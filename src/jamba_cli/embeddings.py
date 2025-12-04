
"""Sentence-transformer wrapper with first-run messaging."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, TYPE_CHECKING

import numpy as np
from rich.console import Console  # type: ignore[import-not-found]

from .settings import EMBEDDING_MODEL_NAME

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import SentenceTransformer


@dataclass(slots=True)
class EmbeddingClient:
    """Lazily loads the embedding model used for RAG."""

    console: Console | None = None
    _model: "SentenceTransformer | None" = field(init=False, default=None, repr=False)

    def _ensure_model(self) -> "SentenceTransformer":
        if self._model is not None:
            return self._model

        if self.console:
            self.console.print(
                "[meta]Preparing sentence-transformer encoder "
                "(first run downloads ~80MB)…[/meta]"
            )

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "sentence-transformers is required for --rag. "
                "Install optional dependencies with `pip install sentence-transformers`."
            ) from exc

        self._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return self._model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return float32 embeddings for the provided texts."""
        if not texts:
            return np.zeros((0, 0), dtype="float32")
        model = self._ensure_model()
        vectors = model.encode(
            list(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return np.asarray(vectors, dtype="float32")

