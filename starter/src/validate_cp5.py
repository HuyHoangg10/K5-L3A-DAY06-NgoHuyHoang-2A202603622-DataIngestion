"""Validate CP5 traces, metrics, hashes, filters, metadata, and audit arithmetic."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .evaluation import summarize


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "starter" / "results" / "cp5"


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    traces = jsonl(RESULTS / "retrieval_traces.local.jsonl")
    index = jsonl(RESULTS / "retrieval_trace_index.jsonl")
    metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    audit = json.loads((RESULTS / "manual_audit.json").read_text(encoding="utf-8"))
    metadata = jsonl(RESULTS / "metadata_audit.jsonl")

    if len(traces) != 60 or sum(len(trace["results"]) for trace in traces) != 180:
        raise ValueError("expected 60 strategy-question traces and 180 results")
    if len(index) != len(traces):
        raise ValueError("trace index length differs from full trace")
    for full, compact in zip(traces, index):
        if (full["strategy"], full["question_id"]) != (
            compact["strategy"],
            compact["question_id"],
        ):
            raise ValueError("trace index order/key mismatch")
        for full_result, compact_result in zip(full["results"], compact["results"]):
            digest = hashlib.sha256(full_result["text"].encode("utf-8")).hexdigest()
            if compact_result["text_sha256"] != digest:
                raise ValueError("trace text hash mismatch")

    if summarize(traces) != metrics:
        raise ValueError("metrics.json does not equal recomputed metrics")
    key_counts = Counter((trace["strategy"], trace["split"]) for trace in traces)
    if key_counts != Counter(
        {("A", "dev"): 15, ("A", "test"): 5, ("B", "dev"): 15,
         ("B", "test"): 5, ("C", "dev"): 15, ("C", "test"): 5}
    ):
        raise ValueError(f"unexpected strategy/split counts: {key_counts}")

    c_filtered = [
        trace for trace in traces if trace["strategy"] == "C" and trace["applied_filter"]
    ]
    if len(c_filtered) != 4:
        raise ValueError("strategy C should apply exactly four supplied filters")
    for trace in c_filtered:
        for result in trace["results"]:
            if not all(
                result["metadata"].get(key) == value
                for key, value in trace["applied_filter"].items()
            ):
                raise ValueError(f"filter violation in {trace['question_id']}")
    if len(metadata) != 45 or not all(row["complete_and_correct"] for row in metadata):
        raise ValueError("metadata audit must contain 45 complete/correct records")

    coherence = audit["coherence"]["records"]
    grounding = audit["grounding"]["records"]
    if len(coherence) != 20 or len(grounding) != 15:
        raise ValueError("manual audit record count mismatch")
    coherence_sums: dict[str, int] = defaultdict(int)
    grounding_sums: dict[str, int] = defaultdict(int)
    for row in coherence:
        coherence_sums[row["strategy"]] += row["score"]
    for row in grounding:
        grounding_sums[row["strategy"]] += row["score"]
    if dict(coherence_sums) != {"A": 10, "B": 20}:
        raise ValueError(f"coherence score arithmetic mismatch: {dict(coherence_sums)}")
    if dict(grounding_sums) != {"A": 7, "B": 7, "C": 7}:
        raise ValueError(f"grounding score arithmetic mismatch: {dict(grounding_sums)}")
    if audit["strategy_selection"]["selected_before_test_interpretation"] != "A":
        raise ValueError("strategy selection does not follow frozen dev rule")

    print("CP5 validation PASSED")
    print("traces: 60 strategy-question rows, 180 top-3 results")
    print("metrics: recomputed exactly")
    print("metadata: 45/45 complete and correct")
    print("filters: 4/4 questions and 12/12 results compliant for C")
    print("manual audit arithmetic: coherence=20 records, grounding=15 records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
