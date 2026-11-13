"""Tests for strict evidence-based retrieval metrics."""

from __future__ import annotations

import unittest

from starter.src.evaluation import evaluate_query, summarize


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.question = {
            "id": "q-test",
            "type": "direct",
            "split": "dev",
            "question": "What is the value?",
            "expected_answer": "42",
            "gold_source_id": "doc-1",
            "gold_page": 2,
            "evidence_quote": "The answer is 42.",
        }

    def result(self, text: str, document_id: str = "doc-1", page: int = 2) -> dict:
        return {
            "rank": 1,
            "chunk_id": "chunk-1",
            "text": text,
            "score": 0.9,
            "document_id": document_id,
            "source": "source",
            "page": page,
            "metadata": {},
        }

    def test_hit_requires_source_page_and_full_evidence(self) -> None:
        hit = evaluate_query("A", self.question, [self.result("The answer\nis 42.")], None)
        self.assertTrue(hit["hit_at_1"])
        wrong_page = evaluate_query("A", self.question, [self.result("The answer is 42.", page=3)], None)
        self.assertFalse(wrong_page["hit_at_1"])
        partial = evaluate_query("A", self.question, [self.result("The answer is")], None)
        self.assertFalse(partial["evidence_hit_at_3"])

    def test_summary_separates_strategy_and_split(self) -> None:
        rows = [
            {"strategy": "A", "split": "dev", "hit_at_1": True, "evidence_hit_at_3": True},
            {"strategy": "A", "split": "dev", "hit_at_1": False, "evidence_hit_at_3": True},
        ]
        summary = summarize(rows)
        self.assertEqual(summary["A"]["dev"]["hit_at_1"]["hits"], 1)
        self.assertEqual(summary["A"]["dev"]["evidence_hit_at_3"]["hits"], 2)


if __name__ == "__main__":
    unittest.main()
