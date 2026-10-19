"""Local sentence-transformer embeddings with explicit model caching."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol


class Embedder(Protocol):
    """Embedding interface for production and deterministic tests."""

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input text."""


class LocalEmbedder:
    """Load all-MiniLM-L6-v2 from a local cache for offline inference."""

    def __init__(
        self,
        cache_dir: Path,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        revision: str | None = None,
        local_files_only: bool = True,
    ) -> None:
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.revision = revision
        self.local_files_only = local_files_only
        self._model: object | None = None

    def load(self) -> None:
        """Load once; download only when local_files_only is explicitly False."""
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._model = SentenceTransformer(
                self.model_name,
                revision=self.revision,
                cache_folder=str(self.cache_dir),
                local_files_only=self.local_files_only,
                trust_remote_code=False,
            )
        except (OSError, ValueError) as exc:
            if self.local_files_only:
                raise RuntimeError("Embedding model is not cached; run `make models`") from exc
            raise

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages locally and normalize vectors for cosine-style search."""
        if not texts:
            return []
        self.load()
        from sentence_transformers import SentenceTransformer

        assert isinstance(self._model, SentenceTransformer)
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return [[float(value) for value in row] for row in vectors]
