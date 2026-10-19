"""Validated retrieval result."""

from pydantic import BaseModel

from repopulse.ingestion.models import Chunk


class SearchHit(BaseModel):
    """One passage with vector distance and optional reranker score."""

    chunk: Chunk
    distance: float
    rerank_score: float | None = None
