"""Evidence-based retrieval evaluation with preserved per-query traces."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Any


def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def _gold_values(question: dict[str, Any], field: str) -> set[Any]:
    value = question[field]
    if isinstance(value, list):
        return set(value)
    return {value}


def result_evidence_hit(question: dict[str, Any], result: dict[str, Any]) -> dict[str, bool]:
    """Check source/page and full normalized evidence containment."""
    source_match = result.get("document_id") in _gold_values(question, "gold_source_id")
    page_match = result.get("page") in _gold_values(question, "gold_page")
    evidence = normalize_for_match(question["evidence_quote"])
    text = normalize_for_match(result.get("text", ""))
    evidence_match = bool(evidence) and evidence in text
    return {
        "source_match": source_match,
        "page_match": page_match,
        "evidence_match": evidence_match,
        "evidence_hit": source_match and page_match and evidence_match,
    }


def evaluate_query(
    strategy: str,
    question: dict[str, Any],
    retrieved: list[dict[str, Any]],
    applied_filter: dict[str, Any] | None,
) -> dict[str, Any]:
    """Annotate top-k retrieval; manual audit labels remain separate."""
    annotated = [{**result, **result_evidence_hit(question, result)} for result in retrieved]
    return {
        "strategy": strategy,
        "question_id": question["id"],
        "split": question["split"],
        "type": question["type"],
        "question": question["question"],
        "expected_answer": question["expected_answer"],
        "gold_source_id": question["gold_source_id"],
        "gold_page": question["gold_page"],
        "evidence_quote": question["evidence_quote"],
        "requested_filter": question.get("filter"),
        "applied_filter": applied_filter,
        "hit_at_1": bool(annotated and annotated[0]["evidence_hit"]),
        "evidence_hit_at_3": any(result["evidence_hit"] for result in annotated[:3]),
        "results": annotated,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1 and Evidence Hit@3 by strategy and split."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["strategy"], row["split"])].append(row)

    summary: dict[str, Any] = {}
    for (strategy, split), group in sorted(grouped.items()):
        total = len(group)
        hit1 = sum(bool(row["hit_at_1"]) for row in group)
        hit3 = sum(bool(row["evidence_hit_at_3"]) for row in group)
        summary.setdefault(strategy, {})[split] = {
            "questions": total,
            "hit_at_1": {
                "hits": hit1,
                "total": total,
                "rate": round(hit1 / total, 4) if total else 0,
            },
            "evidence_hit_at_3": {
                "hits": hit3,
                "total": total,
                "rate": round(hit3 / total, 4) if total else 0,
            },
        }
    return summary
