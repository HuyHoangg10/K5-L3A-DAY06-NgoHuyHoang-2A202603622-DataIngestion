"""Run checkpoint 3: extract, normalize, validate, and create two chunk sets."""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .chunking import by_paragraph, fixed_size
from .ingestion import load_pdf
from .models import Chunk, DocumentPage


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.json"
OUTPUT_DIR = REPO_ROOT / "starter" / "results" / "cp3"
FIXED_SIZE = 500
FIXED_OVERLAP = 50
PARAGRAPH_MAX_SIZE = 1000


def expand_ranges(ranges: list[str]) -> set[int]:
    pages: set[int] = set()
    for value in ranges:
        first, separator, last = value.partition("-")
        start = int(first)
        end = int(last) if separator else start
        pages.update(range(start, end + 1))
    return pages


def compact_length(text: str) -> int:
    """Length under CP1's whitespace-collapse counting rule."""
    return len(re.sub(r"\s+", " ", text).strip())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def document_metadata(manifest: dict[str, Any], document: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": document["title"],
        "year": document["year"],
        "venue": document["venue"],
        "method": document["method"],
        "document_type": document["document_type"],
        "language": manifest["language"],
        "corpus_id": manifest["corpus_id"],
    }


def chunk_stats(chunks: list[Chunk]) -> dict[str, Any]:
    lengths = [len(chunk.text) for chunk in chunks]
    return {
        "chunk_count": len(chunks),
        "characters_total_with_overlap": sum(lengths),
        "characters_min": min(lengths) if lengths else 0,
        "characters_max": max(lengths) if lengths else 0,
        "characters_mean": round(statistics.mean(lengths), 2) if lengths else 0,
        "characters_median": round(statistics.median(lengths), 2) if lengths else 0,
        "characters_p95": sorted(lengths)[max(0, int(0.95 * len(lengths)) - 1)] if lengths else 0,
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def validate_chunks(pages: list[DocumentPage], chunks: list[Chunk], strategy: str, limit: int) -> None:
    page_keys = {(page.document_id, page.page) for page in pages}
    ids: set[str] = set()
    required_metadata = {
        "title",
        "year",
        "venue",
        "method",
        "document_type",
        "language",
        "corpus_id",
        "strategy",
        "chunk_index",
        "char_start",
        "char_end",
    }
    for chunk in chunks:
        if chunk.chunk_id in ids:
            raise ValueError(f"duplicate chunk ID: {chunk.chunk_id}")
        ids.add(chunk.chunk_id)
        if not chunk.text.strip():
            raise ValueError(f"empty chunk: {chunk.chunk_id}")
        if len(chunk.text) > limit:
            raise ValueError(f"oversized chunk {chunk.chunk_id}: {len(chunk.text)} > {limit}")
        if (chunk.document_id, chunk.page) not in page_keys:
            raise ValueError(f"chunk points outside selected pages: {chunk.chunk_id}")
        if chunk.metadata.get("strategy") != strategy:
            raise ValueError(f"wrong strategy metadata: {chunk.chunk_id}")
        missing = required_metadata - chunk.metadata.keys()
        if missing:
            raise ValueError(f"missing metadata {sorted(missing)}: {chunk.chunk_id}")


def main() -> int:
    started = time.perf_counter()
    manifest: dict[str, Any] = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    extraction_issues: list[dict[str, Any]] = []
    selected_pages: list[DocumentPage] = []
    page_audit: list[dict[str, Any]] = []
    document_summaries: list[dict[str, Any]] = []

    for document in manifest["documents"]:
        pdf_path = REPO_ROOT / document["local_path"]
        actual_hash = sha256(pdf_path)
        if actual_hash != document["sha256"]:
            raise ValueError(
                f"SHA-256 mismatch for {document['source_id']}: "
                f"expected {document['sha256']}, got {actual_hash}"
            )

        issues_before = len(extraction_issues)
        all_pages = load_pdf(
            pdf_path,
            document_id=document["source_id"],
            source=document["source_path_or_url"],
            metadata=document_metadata(manifest, document),
            issues=extraction_issues,
        )
        if len(all_pages) != document["physical_pages"]:
            raise ValueError(
                f"physical page mismatch for {document['source_id']}: "
                f"manifest={document['physical_pages']}, PDF={len(all_pages)}"
            )

        included = expand_ranges(document["included_page_ranges"])
        selected = [page for page in all_pages if page.page in included]
        if len(selected) != document["included_pages_count"]:
            raise ValueError(
                f"included page mismatch for {document['source_id']}: "
                f"manifest={document['included_pages_count']}, loaded={len(selected)}"
            )
        empty_selected = [page.page for page in selected if not page.text]
        if empty_selected:
            raise ValueError(
                f"included pages have no extractable text for {document['source_id']}: {empty_selected}"
            )

        compact_characters = sum(compact_length(page.text) for page in selected)
        if compact_characters != document["included_characters_normalized"]:
            raise ValueError(
                f"normalized character mismatch for {document['source_id']}: "
                f"manifest={document['included_characters_normalized']}, CP3={compact_characters}"
            )

        selected_pages.extend(selected)
        document_summaries.append(
            {
                "document_id": document["source_id"],
                "physical_pages": len(all_pages),
                "included_pages": len(selected),
                "excluded_pages": len(all_pages) - len(selected),
                "normalized_characters": compact_characters,
                "extraction_issues": len(extraction_issues) - issues_before,
                "sha256_verified": True,
            }
        )
        for page in all_pages:
            page_audit.append(
                {
                    "document_id": page.document_id,
                    "page": page.page,
                    "included": page.page in included,
                    "status": page.metadata["extraction_status"],
                    "normalized_characters": compact_length(page.text),
                }
            )

    if len(selected_pages) != manifest["included_pages_total"]:
        raise ValueError("corpus included-page total does not match manifest")
    corpus_characters = sum(compact_length(page.text) for page in selected_pages)
    if corpus_characters != manifest["included_characters_normalized_total"]:
        raise ValueError("corpus character total does not match manifest")

    chunks_a = [
        chunk
        for page in selected_pages
        for chunk in fixed_size(page, size=FIXED_SIZE, overlap=FIXED_OVERLAP)
    ]
    chunks_b = [
        chunk
        for page in selected_pages
        for chunk in by_paragraph(page, max_size=PARAGRAPH_MAX_SIZE)
    ]
    validate_chunks(selected_pages, chunks_a, strategy="A", limit=FIXED_SIZE)
    validate_chunks(selected_pages, chunks_b, strategy="B", limit=PARAGRAPH_MAX_SIZE)

    write_jsonl(OUTPUT_DIR / "pages.local.jsonl", (asdict(page) for page in selected_pages))
    write_jsonl(OUTPUT_DIR / "chunks_A.local.jsonl", (asdict(chunk) for chunk in chunks_a))
    write_jsonl(OUTPUT_DIR / "chunks_B.local.jsonl", (asdict(chunk) for chunk in chunks_b))
    write_jsonl(OUTPUT_DIR / "page_audit.jsonl", page_audit)
    (OUTPUT_DIR / "extraction_issues.json").write_text(
        json.dumps(extraction_issues, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - started
    summary = {
        "checkpoint": "CP3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_id": manifest["corpus_id"],
        "normalization": {
            "unicode": "NFC",
            "whitespace": "collapse horizontal whitespace; preserve inferred paragraph boundaries",
            "dehyphenation": False,
            "reason": "avoid changing technical terms without layout/OCR evidence",
        },
        "documents": document_summaries,
        "totals": {
            "physical_pages": sum(item["physical_pages"] for item in document_summaries),
            "included_pages": len(selected_pages),
            "excluded_pages": sum(item["excluded_pages"] for item in document_summaries),
            "normalized_characters": corpus_characters,
            "empty_or_error_pages": len(extraction_issues),
            "elapsed_seconds": round(elapsed, 3),
            "cost_usd": 0,
        },
        "strategy_A": {
            "description": "fixed-size page-scoped character windows",
            "size": FIXED_SIZE,
            "overlap": FIXED_OVERLAP,
            **chunk_stats(chunks_a),
        },
        "strategy_B": {
            "description": "structure-aware paragraph/sentence grouping, page-scoped",
            "max_size": PARAGRAPH_MAX_SIZE,
            "overlap": 0,
            **chunk_stats(chunks_b),
        },
        "strategy_C_plan": {
            "chunking": "identical to strategy B",
            "only_change": "apply trusted manifest metadata filters when the question supplies a filter",
        },
        "metadata_fields": {
            "document_id": "manifest source_id",
            "source": "manifest source_path_or_url",
            "page": "1-based physical PDF page",
            "year": "manifest year; filterable",
            "venue": "manifest venue; filterable",
            "method": "manifest method; filterable",
            "document_type": "manifest document_type; filterable",
            "language": "manifest corpus language; filterable",
        },
        "validation": {
            "pdf_hashes": "passed",
            "physical_page_counts": "passed",
            "included_page_counts": "passed",
            "normalized_character_counts": "passed",
            "selected_pages_nonempty": "passed",
            "chunk_ids_unique": "passed",
            "chunk_size_limits": "passed",
            "traceability_metadata": "passed",
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("CP3 validation PASSED")
    print(
        f"corpus: documents={len(document_summaries)}, physical_pages={summary['totals']['physical_pages']}, "
        f"included_pages={summary['totals']['included_pages']}, "
        f"normalized_characters={summary['totals']['normalized_characters']}"
    )
    print(f"extraction issues: {len(extraction_issues)}")
    print(f"strategy A chunks: {len(chunks_a)} (size={FIXED_SIZE}, overlap={FIXED_OVERLAP})")
    print(f"strategy B chunks: {len(chunks_b)} (max_size={PARAGRAPH_MAX_SIZE})")
    print(f"summary: {OUTPUT_DIR / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
