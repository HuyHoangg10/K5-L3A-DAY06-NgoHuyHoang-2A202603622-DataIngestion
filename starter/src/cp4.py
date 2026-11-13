"""Run checkpoint 4: real embeddings, persistent vector indexes, and top-3 smoke search."""

from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import fields
from datetime import datetime, timezone
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

import sentence_transformers
import torch

from .embeddings import MyEmbedder
from .models import Chunk
from .store import QdrantVectorStore


REPO_ROOT = Path(__file__).resolve().parents[2]
CP3_DIR = REPO_ROOT / "starter" / "results" / "cp3"
OUTPUT_DIR = REPO_ROOT / "starter" / "results" / "cp4"
STORE_PATH = REPO_ROOT / "starter" / "qdrant_storage"
MODEL_CACHE = Path(
    os.environ.get("MODEL_CACHE")
    or os.environ.get("HF_HOME")
    or REPO_ROOT / ".model_cache"
)
MODEL_ID = "google/embeddinggemma-2"
MODEL_REPOSITORY = MODEL_CACHE / "hub" / "models--google--embeddinggemma-2"
BATCH_SIZE = 8


def resolve_snapshot() -> Path:
    refs_main = MODEL_REPOSITORY / "refs" / "main"
    if refs_main.is_file():
        revision = refs_main.read_text(encoding="utf-8").strip()
        candidate = MODEL_REPOSITORY / "snapshots" / revision
        if candidate.is_dir():
            return candidate
    snapshots = sorted((MODEL_REPOSITORY / "snapshots").glob("*"))
    if len(snapshots) != 1:
        raise FileNotFoundError(
            f"cannot resolve a unique local snapshot under {MODEL_REPOSITORY / 'snapshots'}"
        )
    return snapshots[0]


def load_chunks(path: Path) -> list[Chunk]:
    field_names = {field.name for field in fields(Chunk)}
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            unknown = set(value) - field_names
            if unknown:
                raise ValueError(f"{path}:{line_number}: unknown fields {sorted(unknown)}")
            chunks.append(Chunk(**value))
    if not chunks:
        raise ValueError(f"no chunks found in {path}")
    return chunks


def package_version(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


def trace_without_full_vectors(
    strategy: str,
    question: str,
    filters: dict[str, Any] | None,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "question": question,
        "filter": filters,
        "top_k": len(results),
        "results": results,
    }


def main() -> int:
    started_total = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(MODEL_CACHE)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    chunks_a = load_chunks(CP3_DIR / "chunks_A.local.jsonl")
    chunks_b = load_chunks(CP3_DIR / "chunks_B.local.jsonl")
    model_snapshot = resolve_snapshot()

    started_model = time.perf_counter()
    embedder = MyEmbedder(
        MODEL_ID,
        model_path=model_snapshot,
        cache_folder=MODEL_CACHE,
        batch_size=BATCH_SIZE,
        local_files_only=True,
    )
    model_load_seconds = time.perf_counter() - started_model

    started_a = time.perf_counter()
    vectors_a = embedder.embed_documents(
        [chunk.text for chunk in chunks_a],
        [str(chunk.metadata["title"]) for chunk in chunks_a],
    )
    embed_a_seconds = time.perf_counter() - started_a

    started_b = time.perf_counter()
    vectors_b = embedder.embed_documents(
        [chunk.text for chunk in chunks_b],
        [str(chunk.metadata["title"]) for chunk in chunks_b],
    )
    embed_b_seconds = time.perf_counter() - started_b

    store = QdrantVectorStore(STORE_PATH, embedder.dimensions)
    started_store = time.perf_counter()
    sync_first = {
        "A": store.sync("A", chunks_a, vectors_a),
        "B": store.sync("B", chunks_b, vectors_b),
        # C deliberately reuses the exact B chunks and vectors. Only retrieval-time
        # metadata filtering differs, which isolates the factor under study.
        "C": store.sync("C", chunks_b, vectors_b),
    }
    store_first_seconds = time.perf_counter() - started_store
    counts_first = {strategy: store.count(strategy) for strategy in ("A", "B", "C")}
    store.close()

    # Reload the persistent path, sync the same stable IDs again, and verify counts
    # remain unchanged. This is the explicit no-duplication/reload check.
    store = QdrantVectorStore(STORE_PATH, embedder.dimensions)
    started_reload = time.perf_counter()
    sync_second = {
        "A": store.sync("A", chunks_a, vectors_a),
        "B": store.sync("B", chunks_b, vectors_b),
        "C": store.sync("C", chunks_b, vectors_b),
    }
    reload_sync_seconds = time.perf_counter() - started_reload
    counts_reloaded = {strategy: store.count(strategy) for strategy in ("A", "B", "C")}
    expected_counts = {"A": len(chunks_a), "B": len(chunks_b), "C": len(chunks_b)}
    if counts_first != expected_counts or counts_reloaded != expected_counts:
        raise ValueError(
            f"persistent count mismatch: expected={expected_counts}, "
            f"first={counts_first}, reloaded={counts_reloaded}"
        )

    smoke_question = (
        "According to the NAACL-HLT 2019 paper, how is a token's input representation "
        "constructed in BERT?"
    )
    smoke_filter = {"venue": "NAACL-HLT 2019"}
    query_vector = embedder.embed_query(smoke_question)
    traces: list[dict[str, Any]] = []
    for strategy in ("A", "B"):
        results = store.search(strategy, query_vector, top_k=3)
        if len(results) != 3:
            raise ValueError(f"strategy {strategy} did not return top-3")
        traces.append(trace_without_full_vectors(strategy, smoke_question, None, results))
    results_c = store.search("C", query_vector, top_k=3, filters=smoke_filter)
    if len(results_c) != 3:
        raise ValueError("strategy C did not return top-3")
    if any(result["metadata"].get("venue") != smoke_filter["venue"] for result in results_c):
        raise ValueError("strategy C returned a result that violates its venue filter")
    traces.append(trace_without_full_vectors("C", smoke_question, smoke_filter, results_c))
    store.close()

    (OUTPUT_DIR / "smoke_top3.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    elapsed_total = time.perf_counter() - started_total
    summary = {
        "checkpoint": "CP4",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "vector_store": {
            "name": "Qdrant local persistent mode",
            "client_version": distribution_version("qdrant-client"),
            "path": str(STORE_PATH.relative_to(REPO_ROOT)),
            "distance": "cosine",
            "search_mode": "exact local search",
            "collections": ["day6_a", "day6_b", "day6_c"],
        },
        "embedding": {
            "provider": "local SentenceTransformers",
            "model_id": MODEL_ID,
            "model_revision": model_snapshot.name,
            "dimensions": embedder.dimensions,
            "dtype": "float32",
            "device": embedder.device,
            "normalized": True,
            "query_prompt": "QuestionAnswering",
            "document_format": "title: {manifest title} | text: {chunk text}",
            "batch_size": BATCH_SIZE,
            "text_only": True,
            "api_cost_usd": 0,
        },
        "strategies": {
            "A": {"chunks": len(chunks_a), "filter": False},
            "B": {"chunks": len(chunks_b), "filter": False},
            "C": {
                "chunks": len(chunks_b),
                "same_chunks_and_vectors_as": "B",
                "filter": "only when supplied by question/context",
            },
        },
        "timing_seconds": {
            "model_load": round(model_load_seconds, 3),
            "embed_A": round(embed_a_seconds, 3),
            "embed_B_and_reuse_for_C": round(embed_b_seconds, 3),
            "first_store_sync": round(store_first_seconds, 3),
            "reload_and_idempotent_sync": round(reload_sync_seconds, 3),
            "total": round(elapsed_total, 3),
        },
        "persistence_check": {
            "first_sync": sync_first,
            "counts_before_close": counts_first,
            "second_sync_after_reload": sync_second,
            "counts_after_reload": counts_reloaded,
            "expected_counts": expected_counts,
            "no_duplicates": counts_reloaded == expected_counts,
        },
        "smoke_search": {
            "question": smoke_question,
            "top_k": 3,
            "strategies_returning_top3": ["A", "B", "C"],
            "strategy_C_filter": smoke_filter,
            "strategy_C_filter_compliance": "3/3",
            "trace": "starter/results/cp4/smoke_top3.json",
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sentence_transformers": package_version(sentence_transformers),
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("CP4 validation PASSED")
    print(
        f"embedding: {MODEL_ID}@{model_snapshot.name}, "
        f"dimensions={embedder.dimensions}, device={embedder.device}, cost=$0"
    )
    print(f"persistent counts: {counts_reloaded}")
    print("top-3 smoke search: A=3, B=3, C=3; C filter compliance=3/3")
    print(f"summary: {OUTPUT_DIR / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
