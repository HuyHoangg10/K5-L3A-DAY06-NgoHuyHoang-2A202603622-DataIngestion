"""Tests for persistent vector sync, cosine search, and metadata filters."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from starter.src.models import Chunk
from starter.src.store import QdrantVectorStore


class QdrantVectorStoreTests(unittest.TestCase):
    def make_chunk(self, chunk_id: str, year: int, text: str) -> Chunk:
        return Chunk(
            chunk_id=chunk_id,
            document_id=f"doc-{year}",
            source=f"https://example.test/{year}.pdf",
            page=1,
            text=text,
            metadata={
                "year": year,
                "venue": f"Venue {year}",
                "method": "example",
                "document_type": "research_paper",
                "language": "en",
            },
        )

    def test_persistence_idempotency_search_filter_and_stale_delete(self) -> None:
        test_temp_root = Path(__file__).resolve().parents[1] / "results"
        test_temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_temp_root) as directory:
            path = Path(directory) / "qdrant"
            chunks = [
                self.make_chunk("a", 2019, "first"),
                self.make_chunk("b", 2020, "second"),
            ]
            vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            store = QdrantVectorStore(path, dimensions=3, prefix="test")
            self.assertEqual(store.sync("A", chunks, vectors)["after"], 2)
            self.assertEqual(store.sync("A", chunks, vectors)["after"], 2)
            result = store.search("A", [1.0, 0.0, 0.0], top_k=1)
            self.assertEqual(result[0]["chunk_id"], "a")
            filtered = store.search(
                "A", [1.0, 0.0, 0.0], top_k=2, filters={"year": 2020}
            )
            self.assertEqual([item["chunk_id"] for item in filtered], ["b"])
            store.close()

            reloaded = QdrantVectorStore(path, dimensions=3, prefix="test")
            self.assertEqual(reloaded.count("A"), 2)
            sync = reloaded.sync("A", chunks[:1], vectors[:1])
            self.assertEqual(sync["deleted_stale"], 1)
            self.assertEqual(reloaded.count("A"), 1)
            with self.assertRaises(ValueError):
                reloaded.search("A", [1.0, 0.0, 0.0], filters={"chunk_id": "a"})
            reloaded.close()


if __name__ == "__main__":
    unittest.main()
