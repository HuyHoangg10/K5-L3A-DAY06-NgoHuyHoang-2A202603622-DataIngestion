"""Persistent Qdrant vector store with cosine search and metadata filters."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Iterable

from qdrant_client import QdrantClient, models

from .models import Chunk


class QdrantVectorStore:
    """Qdrant local-mode store; the on-disk path is reloadable and filterable."""

    FILTER_FIELDS = {"year", "venue", "method", "document_type", "language"}

    def __init__(self, path: Path | str, dimensions: int, prefix: str = "day6") -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.dimensions = dimensions
        self.prefix = prefix
        self.client = QdrantClient(path=str(self.path))

    def close(self) -> None:
        self.client.close()

    def collection_name(self, strategy: str) -> str:
        normalized = strategy.strip().lower()
        if normalized not in {"a", "b", "c"}:
            raise ValueError(f"unknown strategy: {strategy!r}")
        return f"{self.prefix}_{normalized}"

    def create_schema(self, strategy: str) -> None:
        """Create a cosine collection or validate its existing dimensions."""
        name = self.collection_name(strategy)
        if not self.client.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=self.dimensions,
                    distance=models.Distance.COSINE,
                ),
            )
            return

        info = self.client.get_collection(name)
        params = info.config.params.vectors
        if not isinstance(params, models.VectorParams):
            raise ValueError(f"collection {name} does not use a single vector configuration")
        if params.size != self.dimensions or params.distance != models.Distance.COSINE:
            raise ValueError(
                f"collection {name} is incompatible: size={params.size}, distance={params.distance}"
            )

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    @staticmethod
    def _batches(values: list[Any], size: int = 64) -> Iterable[list[Any]]:
        for start in range(0, len(values), size):
            yield values[start : start + size]

    def sync(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> dict[str, int]:
        """Upsert stable IDs and delete stale points from the strategy collection."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have equal length")
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("chunk IDs must be unique")
        for index, vector in enumerate(vectors):
            if len(vector) != self.dimensions:
                raise ValueError(
                    f"vector {index} has {len(vector)} dimensions; expected {self.dimensions}"
                )

        self.create_schema(strategy)
        name = self.collection_name(strategy)
        points: list[models.PointStruct] = []
        desired_ids: set[str] = set()
        for chunk, vector in zip(chunks, vectors):
            point_id = self._point_id(chunk.chunk_id)
            desired_ids.add(point_id)
            payload = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "source": chunk.source,
                "page": chunk.page,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            for field in self.FILTER_FIELDS:
                if field in chunk.metadata:
                    payload[field] = chunk.metadata[field]
            points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))

        existing_ids: set[str] = set()
        offset: Any = None
        while True:
            records, offset = self.client.scroll(
                collection_name=name,
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            existing_ids.update(str(record.id) for record in records)
            if offset is None:
                break

        stale_ids = existing_ids - desired_ids
        if stale_ids:
            self.client.delete(
                collection_name=name,
                points_selector=models.PointIdsList(points=sorted(stale_ids)),
                wait=True,
            )
        for batch in self._batches(points):
            self.client.upsert(collection_name=name, points=batch, wait=True)

        return {
            "before": len(existing_ids),
            "upserted": len(points),
            "deleted_stale": len(stale_ids),
            "after": self.count(strategy),
        }

    def count(self, strategy: str) -> int:
        return int(
            self.client.count(
                collection_name=self.collection_name(strategy), exact=True
            ).count
        )

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return cosine-ranked chunks, applying only allow-listed metadata filters."""
        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"query vector has {len(query_vector)} dimensions; expected {self.dimensions}"
            )
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        conditions: list[models.FieldCondition] = []
        for key, value in (filters or {}).items():
            if key not in self.FILTER_FIELDS:
                raise ValueError(f"filter field is not allowed: {key!r}")
            if not isinstance(value, (str, int, bool)):
                raise ValueError(f"unsupported filter value for {key!r}: {value!r}")
            conditions.append(
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
            )
        query_filter = models.Filter(must=conditions) if conditions else None
        response = self.client.query_points(
            collection_name=self.collection_name(strategy),
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        results: list[dict[str, Any]] = []
        for rank, point in enumerate(response.points, start=1):
            payload = point.payload or {}
            results.append(
                {
                    "rank": rank,
                    "chunk_id": payload.get("chunk_id"),
                    "text": payload.get("text", ""),
                    "score": float(point.score),
                    "document_id": payload.get("document_id"),
                    "source": payload.get("source"),
                    "page": payload.get("page"),
                    "metadata": payload.get("metadata", {}),
                }
            )
        return results
