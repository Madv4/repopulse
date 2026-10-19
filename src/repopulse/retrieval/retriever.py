"""Document-level retrieval with optional second-stage reranking."""

from repopulse.ingestion.embed import Embedder
from repopulse.ingestion.store import ChromaStore
from repopulse.retrieval.models import SearchHit
from repopulse.retrieval.rerank import Reranker


class Retriever:
    """Search passages and return at most one top hit per source document."""

    def __init__(
        self,
        store: ChromaStore,
        embedder: Embedder,
        reranker: Reranker | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker

    def search(self, query: str, *, repo: str, k: int = 5, rerank: bool = False) -> list[SearchHit]:
        """Retrieve a candidate pool, optionally rerank, then deduplicate sources."""
        if k < 1:
            raise ValueError("k must be positive")
        if rerank and self.reranker is None:
            raise ValueError("rerank=True requires a reranker")
        candidates = self.store.search(query, self.embedder, repo=repo, limit=max(20, k * 4))
        if rerank and self.reranker is not None and candidates:
            scores = self.reranker.score(query, [hit.chunk.text for hit in candidates])
            if len(scores) != len(candidates):
                raise ValueError("reranker returned the wrong number of scores")
            for hit, score in zip(candidates, scores, strict=True):
                hit.rerank_score = score
            candidates.sort(
                key=lambda hit: hit.rerank_score if hit.rerank_score is not None else float("-inf"),
                reverse=True,
            )
        seen: set[str] = set()
        distinct: list[SearchHit] = []
        for hit in candidates:
            if hit.chunk.source_id not in seen:
                seen.add(hit.chunk.source_id)
                distinct.append(hit)
            if len(distinct) == k:
                break
        return distinct
