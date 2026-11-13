"""Validate CP2 benchmark metadata and gold evidence against the local PDFs."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT / "benchmark" / "questions.json"
MANIFEST_PATH = ROOT / "data" / "manifest.json"
REQUIRED_FIELDS = {
    "id",
    "type",
    "split",
    "question",
    "expected_answer",
    "gold_source_id",
    "gold_page",
    "evidence_quote",
}


def normalize(text: str) -> str:
    """Apply the same conservative whitespace normalization used for evidence."""
    return re.sub(r"\s+", " ", text).strip()


def expand_ranges(ranges: list[str]) -> set[int]:
    pages: set[int] = set()
    for value in ranges:
        start_text, separator, end_text = value.partition("-")
        start = int(start_text)
        end = int(end_text) if separator else start
        pages.update(range(start, end + 1))
    return pages


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def main() -> int:
    questions: list[dict[str, Any]] = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    manifest: dict[str, Any] = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    documents = {document["source_id"]: document for document in manifest["documents"]}
    errors: list[str] = []

    if len(questions) < 20:
        fail(errors, f"benchmark has {len(questions)} questions; at least 20 are required")

    ids = [question.get("id") for question in questions]
    if len(ids) != len(set(ids)):
        fail(errors, "question IDs are not unique")

    split_counts = Counter(question.get("split") for question in questions)
    if split_counts["dev"] < 15 or split_counts["test"] < 5:
        fail(errors, f"invalid split counts: {dict(split_counts)}")
    unexpected_splits = set(split_counts) - {"dev", "test"}
    if unexpected_splits:
        fail(errors, f"unexpected split values: {sorted(unexpected_splits)}")

    filtered = [question for question in questions if "filter" in question]
    filtered_test = [question for question in filtered if question.get("split") == "test"]
    if len(filtered) < 3:
        fail(errors, f"only {len(filtered)} filter questions; at least 3 are required")
    if not filtered_test:
        fail(errors, "at least one test question must contain a filter")

    source_counts: Counter[str] = Counter()
    pdf_cache: dict[str, PdfReader] = {}

    for position, question in enumerate(questions, start=1):
        qid = question.get("id", f"position-{position}")
        missing = REQUIRED_FIELDS - question.keys()
        if missing:
            fail(errors, f"{qid}: missing fields {sorted(missing)}")
            continue

        for field in ("question", "expected_answer", "evidence_quote"):
            if not isinstance(question[field], str) or not question[field].strip():
                fail(errors, f"{qid}: {field} must be a non-empty string")

        source_id = question["gold_source_id"]
        source_counts[source_id] += 1
        document = documents.get(source_id)
        if document is None:
            fail(errors, f"{qid}: unknown gold_source_id {source_id!r}")
            continue

        page = question["gold_page"]
        if not isinstance(page, int) or isinstance(page, bool):
            fail(errors, f"{qid}: gold_page must be an integer")
            continue

        included_pages = expand_ranges(document["included_page_ranges"])
        if page not in included_pages:
            fail(errors, f"{qid}: page {page} is not included for {source_id}")

        pdf_path = ROOT / document["local_path"]
        if not pdf_path.is_file():
            fail(errors, f"{qid}: local PDF is missing: {pdf_path}")
            continue

        reader = pdf_cache.setdefault(source_id, PdfReader(pdf_path))
        if not 1 <= page <= len(reader.pages):
            fail(errors, f"{qid}: page {page} is outside the PDF")
            continue

        page_text = normalize(reader.pages[page - 1].extract_text() or "")
        evidence = normalize(question["evidence_quote"])
        if evidence not in page_text:
            fail(errors, f"{qid}: evidence is not present on {source_id} page {page}")

        metadata_filter = question.get("filter")
        if metadata_filter is not None:
            if not isinstance(metadata_filter, dict) or not metadata_filter:
                fail(errors, f"{qid}: filter must be a non-empty object")
                continue
            for key, value in metadata_filter.items():
                if key not in document:
                    fail(errors, f"{qid}: filter key {key!r} is absent from manifest metadata")
                elif document[key] != value:
                    fail(
                        errors,
                        f"{qid}: filter {key}={value!r} excludes its gold document "
                        f"({document[key]!r})",
                    )
                if str(value).casefold() not in question["question"].casefold():
                    fail(errors, f"{qid}: filter value {value!r} is not stated in the question")

    unknown_sources = set(source_counts) - set(documents)
    if unknown_sources:
        fail(errors, f"unknown sources in benchmark: {sorted(unknown_sources)}")

    if errors:
        print(f"CP2 benchmark validation FAILED with {len(errors)} error(s):")
        for error in errors:
            print(f"- {error}")
        return 1

    print("CP2 benchmark validation PASSED")
    print(f"questions: {len(questions)}")
    print(f"split: dev={split_counts['dev']}, test={split_counts['test']}")
    print(f"filter questions: total={len(filtered)}, test={len(filtered_test)}")
    print("questions per source:")
    for source_id in documents:
        print(f"- {source_id}: {source_counts[source_id]}")
    print(f"validated evidence pages: {len(questions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
