"""Optional local cross-encoder for second-stage ranking."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol


class Reranker(Protocol):
    """Rerank retrieved passages without requiring a hosted API."""

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Return one relevance score per passage."""


class LocalCrossEncoderReranker:
    """Use the local ms-marco MiniLM cross-encoder after vector retrieval."""

    def __init__(
        self,
        cache_dir: Path,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
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
        from sentence_transformers import CrossEncoder

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._model = CrossEncoder(
                self.model_name,
                revision=self.revision,
                cache_folder=str(self.cache_dir),
                local_files_only=self.local_files_only,
                trust_remote_code=False,
            )
        except (OSError, ValueError) as exc:
            if self.local_files_only:
                raise RuntimeError("Reranker model is not cached; run `make models`") from exc
            raise

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Score query/passage pairs entirely on this machine."""
        if not passages:
            return []
        self.load()
        from sentence_transformers import CrossEncoder

        assert isinstance(self._model, CrossEncoder)
        scores = self._model.predict([(query, passage) for passage in passages])
        return [float(value) for value in scores]
