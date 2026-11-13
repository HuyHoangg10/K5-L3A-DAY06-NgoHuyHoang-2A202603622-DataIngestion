"""Unit tests for CP3 normalization and chunk construction."""

from __future__ import annotations

import unittest

from starter.src.chunking import by_paragraph, fixed_size
from starter.src.ingestion import normalize_text
from starter.src.models import DocumentPage


class CP3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = {
            "title": "Example",
            "year": 2020,
            "venue": "ExampleConf",
            "method": "example",
            "document_type": "research_paper",
            "language": "en",
            "corpus_id": "example-corpus",
        }

    def make_page(self, text: str) -> DocumentPage:
        return DocumentPage(
            document_id="doc-1",
            source="https://example.test/paper.pdf",
            page=3,
            text=text,
            metadata=dict(self.metadata),
        )

    def test_normalize_text_is_nfc_and_retains_structure(self) -> None:
        raw = "1 Introduction\nCafe\u0301 is here.\n\nSecond   paragraph.\x00"
        normalized = normalize_text(raw)
        self.assertIn("Café", normalized)
        self.assertIn("\n\n", normalized)
        self.assertNotIn("\x00", normalized)
        self.assertNotIn("   ", normalized)

    def test_fixed_size_is_bounded_traceable_and_deterministic(self) -> None:
        page = self.make_page(" ".join(f"token-{index}" for index in range(150)))
        first = fixed_size(page, size=120, overlap=20)
        second = fixed_size(page, size=120, overlap=20)
        self.assertGreater(len(first), 1)
        self.assertEqual([chunk.chunk_id for chunk in first], [chunk.chunk_id for chunk in second])
        self.assertTrue(all(0 < len(chunk.text) <= 120 for chunk in first))
        self.assertTrue(all(chunk.page == 3 for chunk in first))
        self.assertTrue(all(chunk.metadata["year"] == 2020 for chunk in first))
        self.assertEqual(len(first), len({chunk.chunk_id for chunk in first}))

    def test_structure_aware_chunks_are_bounded_and_keep_paragraphs(self) -> None:
        page = self.make_page(
            "A complete first paragraph with one claim.\n\n"
            "A complete second paragraph with another claim.\n\n"
            + "A long final sentence "
            + "word " * 80
            + "."
        )
        chunks = by_paragraph(page, max_size=140)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(chunk.text) <= 140 for chunk in chunks))
        self.assertTrue(all(chunk.metadata["strategy"] == "B" for chunk in chunks))
        self.assertEqual(len(chunks), len({chunk.chunk_id for chunk in chunks}))

    def test_invalid_chunk_parameters_fail_loudly(self) -> None:
        page = self.make_page("some text")
        with self.assertRaises(ValueError):
            fixed_size(page, size=100, overlap=100)
        with self.assertRaises(ValueError):
            by_paragraph(page, max_size=0)


if __name__ == "__main__":
    unittest.main()
