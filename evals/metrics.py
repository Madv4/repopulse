"""Pure retrieval metrics used by the recorded-fixture evaluation."""

from collections.abc import Sequence

from pydantic import BaseModel, Field


class RetrievalCase(BaseModel):
    """A human-labeled query and one or more relevant source URLs."""

    id: str
    query: str
    relevant_urls: list[str] = Field(min_length=1)


class QueryResult(BaseModel):
    """Observed ranked URLs and per-case metrics."""

    case_id: str
    urls: list[str]
    reciprocal_rank: float
    hit: bool


def measure_case(case: RetrievalCase, urls: Sequence[str], k: int) -> QueryResult:
    """Compute binary hit and reciprocal rank from actual ranked results."""
    if k < 1:
        raise ValueError("k must be positive")
    relevant = set(case.relevant_urls)
    ranked = list(urls[:k])
    rank = next((index for index, url in enumerate(ranked, start=1) if url in relevant), None)
    return QueryResult(
        case_id=case.id,
        urls=ranked,
        reciprocal_rank=1.0 / rank if rank is not None else 0.0,
        hit=rank is not None,
    )


def aggregate(results: Sequence[QueryResult]) -> tuple[float, float]:
    """Return measured hit rate@k and MRR@k for a nonempty case set."""
    if not results:
        raise ValueError("Cannot aggregate an empty evaluation")
    count = len(results)
    return sum(result.hit for result in results) / count, sum(
        result.reciprocal_rank for result in results
    ) / count
