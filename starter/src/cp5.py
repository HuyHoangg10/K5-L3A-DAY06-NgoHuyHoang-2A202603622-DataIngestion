"""Run CP5 retrieval benchmark and prepare fixed manual-audit candidates."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .embeddings import MyEmbedder
from .evaluation import evaluate_query, normalize_for_match, summarize
from .models import Chunk
from .store import QdrantVectorStore


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = REPO_ROOT / "benchmark" / "questions.json"
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.json"
CP3_DIR = REPO_ROOT / "starter" / "results" / "cp3"
OUTPUT_DIR = REPO_ROOT / "starter" / "results" / "cp5"
STORE_PATH = REPO_ROOT / "starter" / "qdrant_storage"
MODEL_CACHE = Path(
    os.environ.get("MODEL_CACHE")
    or os.environ.get("HF_HOME")
    or REPO_ROOT / ".model_cache"
)
MODEL_ID = "google/embeddinggemma-2"
MODEL_REPOSITORY = MODEL_CACHE / "hub" / "models--google--embeddinggemma-2"
BATCH_SIZE = 8

# Frozen before retrieval. These balanced dev evidence positions are independent of
# scores. All five test questions are reserved for the grounding audit.
COHERENCE_QUESTION_IDS = ["q01", "q02", "q03", "q06", "q07", "q08", "q11", "q12", "q16", "q17"]
GROUNDING_QUESTION_IDS = ["q05", "q10", "q15", "q19", "q20"]


def resolve_snapshot() -> Path:
    refs_main = MODEL_REPOSITORY / "refs" / "main"
    if refs_main.is_file():
        revision = refs_main.read_text(encoding="utf-8").strip()
        candidate = MODEL_REPOSITORY / "snapshots" / revision
        if candidate.is_dir():
            return candidate
    snapshots = sorted((MODEL_REPOSITORY / "snapshots").glob("*"))
    if len(snapshots) != 1:
        raise FileNotFoundError("cannot resolve one local EmbeddingGemma 2 snapshot")
    return snapshots[0]


def load_chunks(path: Path) -> list[Chunk]:
    allowed = {field.name for field in fields(Chunk)}
    result: list[Chunk] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if set(value) - allowed:
                raise ValueError(f"unexpected chunk fields in {path}")
            result.append(Chunk(**value))
    return result


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def compact_trace(row: dict[str, Any]) -> dict[str, Any]:
    compact_results = []
    for result in row["results"]:
        compact = {key: value for key, value in result.items() if key != "text"}
        compact["text_sha256"] = hashlib.sha256(result["text"].encode("utf-8")).hexdigest()
        compact["text_characters"] = len(result["text"])
        compact_results.append(compact)
    output = {key: value for key, value in row.items() if key != "results"}
    output["results"] = compact_results
    return output


def evidence_overlap(evidence: str, chunk_text: str) -> float:
    evidence_tokens = normalize_for_match(evidence).casefold().split()
    chunk_tokens = set(normalize_for_match(chunk_text).casefold().split())
    if not evidence_tokens:
        return 0.0
    return sum(token in chunk_tokens for token in evidence_tokens) / len(evidence_tokens)


def select_coherence_candidates(
    questions: list[dict[str, Any]], chunks_by_strategy: dict[str, list[Chunk]]
) -> list[dict[str, Any]]:
    by_id = {question["id"]: question for question in questions}
    candidates: list[dict[str, Any]] = []
    for question_id in COHERENCE_QUESTION_IDS:
        question = by_id[question_id]
        for strategy in ("A", "B"):
            eligible = [
                chunk
                for chunk in chunks_by_strategy[strategy]
                if chunk.document_id == question["gold_source_id"]
                and chunk.page == question["gold_page"]
            ]
            if not eligible:
                raise ValueError(f"no coherence candidate for {question_id}/{strategy}")
            selected = max(
                eligible,
                key=lambda chunk: evidence_overlap(question["evidence_quote"], chunk.text),
            )
            candidates.append(
                {
                    "position_id": question_id,
                    "strategy": strategy,
                    "gold_source_id": question["gold_source_id"],
                    "gold_page": question["gold_page"],
                    "evidence_quote": question["evidence_quote"],
                    "selection_rule": "same source/page; maximum evidence-token overlap; independent of retrieval score",
                    "selection_overlap_ratio": round(
                        evidence_overlap(question["evidence_quote"], selected.text), 4
                    ),
                    "chunk": asdict(selected),
                }
            )
    return candidates


def metadata_audit_records(
    traces: list[dict[str, Any]], manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    documents = {document["source_id"]: document for document in manifest["documents"]}
    records: list[dict[str, Any]] = []
    for trace in traces:
        if trace["split"] != "test":
            continue
        for result in trace["results"]:
            document = documents.get(result["document_id"])
            metadata = result["metadata"]
            checks = {
                "known_document": document is not None,
                "source_correct": bool(document and result["source"] == document["source_path_or_url"]),
                "page_valid": bool(document and 1 <= result["page"] <= document["physical_pages"]),
                "year_correct": bool(document and metadata.get("year") == document["year"]),
                "venue_correct": bool(document and metadata.get("venue") == document["venue"]),
                "method_correct": bool(document and metadata.get("method") == document["method"]),
                "document_type_correct": bool(
                    document and metadata.get("document_type") == document["document_type"]
                ),
                "language_present": metadata.get("language") == manifest["language"],
            }
            records.append(
                {
                    "strategy": trace["strategy"],
                    "question_id": trace["question_id"],
                    "rank": result["rank"],
                    "chunk_id": result["chunk_id"],
                    "document_id": result["document_id"],
                    "page": result["page"],
                    "checks": checks,
                    "complete_and_correct": all(checks.values()),
                    "review_status": "manifest cross-check; included in assisted manual review",
                }
            )
    return records


def filter_summary(traces: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for strategy in ("A", "B", "C"):
        relevant = [
            trace
            for trace in traces
            if trace["strategy"] == strategy and trace["requested_filter"] is not None
        ]
        applied = [trace for trace in relevant if trace["applied_filter"] is not None]
        compliant_questions = 0
        compliant_results = 0
        total_results = 0
        for trace in applied:
            expected = trace["applied_filter"]
            flags = [
                all(result["metadata"].get(key) == value for key, value in expected.items())
                for result in trace["results"]
            ]
            compliant_results += sum(flags)
            total_results += len(flags)
            compliant_questions += int(all(flags) and bool(flags))
        output[strategy] = {
            "questions_requesting_filter": len(relevant),
            "questions_with_filter_applied": len(applied),
            "fully_compliant_questions": compliant_questions,
            "compliant_results": compliant_results,
            "retrieved_results_under_applied_filters": total_results,
            "note": "A/B intentionally do not apply filters; C applies only supplied filters.",
        }
    return output


def main() -> int:
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(MODEL_CACHE)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    questions: list[dict[str, Any]] = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    manifest: dict[str, Any] = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if [question["id"] for question in questions if question["split"] == "test"] != GROUNDING_QUESTION_IDS:
        raise ValueError("grounding audit IDs no longer equal the frozen five-question test split")

    chunks_a = load_chunks(CP3_DIR / "chunks_A.local.jsonl")
    chunks_b = load_chunks(CP3_DIR / "chunks_B.local.jsonl")
    coherence_candidates = select_coherence_candidates(
        questions, {"A": chunks_a, "B": chunks_b}
    )
    write_jsonl(OUTPUT_DIR / "coherence_candidates.local.jsonl", coherence_candidates)

    snapshot = resolve_snapshot()
    embedder = MyEmbedder(
        MODEL_ID,
        model_path=snapshot,
        cache_folder=MODEL_CACHE,
        batch_size=BATCH_SIZE,
        local_files_only=True,
    )
    query_started = time.perf_counter()
    query_vectors = embedder.embed_queries([question["question"] for question in questions])
    query_seconds = time.perf_counter() - query_started

    store = QdrantVectorStore(STORE_PATH, embedder.dimensions)
    expected_counts = {"A": len(chunks_a), "B": len(chunks_b), "C": len(chunks_b)}
    actual_counts = {strategy: store.count(strategy) for strategy in ("A", "B", "C")}
    if actual_counts != expected_counts:
        raise ValueError(f"Qdrant counts changed: expected={expected_counts}, actual={actual_counts}")

    traces: list[dict[str, Any]] = []
    for question, vector in zip(questions, query_vectors):
        for strategy in ("A", "B", "C"):
            applied_filter = question.get("filter") if strategy == "C" else None
            results = store.search(strategy, vector, top_k=3, filters=applied_filter)
            if len(results) != 3:
                raise ValueError(f"{strategy}/{question['id']} returned {len(results)} results")
            traces.append(evaluate_query(strategy, question, results, applied_filter))
    store.close()

    write_jsonl(OUTPUT_DIR / "retrieval_traces.local.jsonl", traces)
    write_jsonl(OUTPUT_DIR / "retrieval_trace_index.jsonl", map(compact_trace, traces))
    grounding_candidates = [
        trace for trace in traces if trace["question_id"] in GROUNDING_QUESTION_IDS
    ]
    write_jsonl(OUTPUT_DIR / "grounding_candidates.local.jsonl", grounding_candidates)

    metadata_records = metadata_audit_records(traces, manifest)
    write_jsonl(OUTPUT_DIR / "metadata_audit.jsonl", metadata_records)
    metrics = summarize(traces)
    filters = filter_summary(traces)
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "checkpoint": "CP5",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frozen_inputs": {
            "benchmark_questions": len(questions),
            "dev": sum(question["split"] == "dev" for question in questions),
            "test": sum(question["split"] == "test" for question in questions),
            "model_id": MODEL_ID,
            "model_revision": snapshot.name,
            "vector_store_counts": actual_counts,
            "top_k": 3,
            "configuration_changed_after_viewing_test": False,
        },
        "retrieval": {
            "queries_embedded_once": len(query_vectors),
            "strategy_question_traces": len(traces),
            "retrieved_results": sum(len(trace["results"]) for trace in traces),
            "query_embedding_seconds": round(query_seconds, 3),
            "total_seconds": round(time.perf_counter() - started, 3),
            "api_cost_usd": 0,
        },
        "metrics": metrics,
        "filter_compliance": filters,
        "metadata_audit_precheck": {
            "records": len(metadata_records),
            "complete_and_correct": sum(
                record["complete_and_correct"] for record in metadata_records
            ),
            "scope": "5 test questions x top-3 x 3 strategies",
        },
        "manual_audit_plan": {
            "reviewer": "student review with AI assistance",
            "coherence_positions": COHERENCE_QUESTION_IDS,
            "coherence_candidates": len(coherence_candidates),
            "grounding_test_questions": GROUNDING_QUESTION_IDS,
            "grounding_strategy_cases": len(grounding_candidates),
            "score_definition": {
                "coherence": "0=cut/incomprehensible, 1=partly understandable, 2=self-contained complete idea",
                "grounding": "0=no evidence, 1=partial/missing context, 2=sufficient and traceable",
            },
        },
        "artifacts": {
            "full_trace_local": "starter/results/cp5/retrieval_traces.local.jsonl",
            "versionable_trace_index": "starter/results/cp5/retrieval_trace_index.jsonl",
            "metrics": "starter/results/cp5/metrics.json",
            "metadata_audit": "starter/results/cp5/metadata_audit.jsonl",
            "coherence_candidates_local": "starter/results/cp5/coherence_candidates.local.jsonl",
            "grounding_candidates_local": "starter/results/cp5/grounding_candidates.local.jsonl",
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("CP5 automated benchmark PASSED")
    for strategy in ("A", "B", "C"):
        dev = metrics[strategy]["dev"]
        test = metrics[strategy]["test"]
        print(
            f"{strategy}: dev Hit@1={dev['hit_at_1']['hits']}/{dev['hit_at_1']['total']}, "
            f"Hit@3={dev['evidence_hit_at_3']['hits']}/{dev['evidence_hit_at_3']['total']}; "
            f"test Hit@1={test['hit_at_1']['hits']}/{test['hit_at_1']['total']}, "
            f"Hit@3={test['evidence_hit_at_3']['hits']}/{test['evidence_hit_at_3']['total']}"
        )
    print(
        f"metadata precheck: {summary['metadata_audit_precheck']['complete_and_correct']}/"
        f"{summary['metadata_audit_precheck']['records']} complete and correct"
    )
    print(f"summary: {OUTPUT_DIR / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
