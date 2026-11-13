"""Local EmbeddingGemma 2 adapter used by every retrieval strategy."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(
        self, texts: list[str], titles: Sequence[str] | None = None
    ) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...

    def embed_queries(self, questions: list[str]) -> list[list[float]]:
        """Return one compatible query vector per question."""
        ...


class MyEmbedder:
    """SentenceTransformers adapter for real, normalized 768d embeddings.

    The model card recommends asymmetric instructions for question answering:
    queries use ``QuestionAnswering`` and documents use
    ``title: {title} | text: {content}``. Vision/audio encoders are disabled because
    this corpus is text-only. No mock or random fallback is provided.
    """

    def __init__(
        self,
        model_name: str,
        *,
        model_path: Path | str | None = None,
        cache_folder: Path | str | None = None,
        batch_size: int = 8,
        local_files_only: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model_name = model_name
        self.model_path = str(model_path or model_name)
        self.model_revision = Path(self.model_path).name if model_path else "resolved-by-cache"
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # The model card explicitly warns against float16. Float32 is safe on CPU
        # and GPU and is chosen here for reproducibility across lab machines.
        self.dtype = torch.float32
        self._model = SentenceTransformer(
            self.model_path,
            cache_folder=str(cache_folder) if cache_folder else None,
            local_files_only=local_files_only,
            device=self.device,
            model_kwargs={"torch_dtype": self.dtype},
            config_kwargs={"vision_config": None, "audio_config": None},
        )
        dimension = self._model.get_embedding_dimension()
        if dimension is None:
            raise RuntimeError("embedding model did not report an output dimension")
        self.dimensions = int(dimension)
        if self.dimensions != 768:
            raise ValueError(f"expected EmbeddingGemma 2 to output 768d, got {self.dimensions}")

    def _validate(self, vectors: np.ndarray, expected_rows: int) -> list[list[float]]:
        if vectors.shape != (expected_rows, self.dimensions):
            raise ValueError(
                f"unexpected embedding shape {vectors.shape}; "
                f"expected ({expected_rows}, {self.dimensions})"
            )
        if not np.isfinite(vectors).all():
            raise ValueError("embedding model returned NaN or infinite values")
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError("embeddings are not L2-normalized")
        return vectors.astype(np.float32, copy=False).tolist()

    def embed_documents(
        self, texts: list[str], titles: Sequence[str] | None = None
    ) -> list[list[float]]:
        if not texts:
            return []
        if titles is not None and len(titles) != len(texts):
            raise ValueError("titles and texts must have equal length")
        if titles is None:
            prepared = [f"title: none | text: {text}" for text in texts]
        else:
            prepared = [f"title: {title} | text: {text}" for title, text in zip(titles, texts)]
        vectors = self._model.encode(
            prepared,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return self._validate(np.asarray(vectors), len(texts))

    def embed_query(self, question: str) -> list[float]:
        if not question.strip():
            raise ValueError("question must not be empty")
        return self.embed_queries([question])[0]

    def embed_queries(self, questions: list[str]) -> list[list[float]]:
        if not questions:
            return []
        if any(not question.strip() for question in questions):
            raise ValueError("questions must not be empty")
        vector = self._model.encode(
            questions,
            prompt_name="QuestionAnswering",
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(questions) > self.batch_size,
        )
        return self._validate(np.asarray(vector), len(questions))
