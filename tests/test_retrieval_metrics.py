"""Deterministic metric and fixture integrity checks."""

import json

import pytest

from evals.metrics import RetrievalCase, aggregate, measure_case
from evals.run_retrieval import DATASET, FIXTURE


def test_hit_rate_and_mrr_are_computed_from_ranks() -> None:
    first = RetrievalCase(id="a", query="a", relevant_urls=["url-a"])
    second = RetrievalCase(id="b", query="b", relevant_urls=["url-b"])
    results = [
        measure_case(first, ["irrelevant", "url-a"], 3),
        measure_case(second, ["irrelevant", "other"], 3),
    ]
    assert aggregate(results) == (0.5, 0.25)
    with pytest.raises(ValueError):
        aggregate([])


def test_every_label_is_in_recorded_snapshot() -> None:
    snapshot = json.loads(FIXTURE.read_text(encoding="utf-8"))
    urls = {item["html_url"] for item in snapshot["issues"]}
    urls.update(item["html_url"] for item in snapshot["pulls"].values())
    urls.update(item["html_url"] for item in snapshot["commits"])
    cases = [
        RetrievalCase.model_validate_json(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(cases) >= 10
    assert len({case.id for case in cases}) == len(cases)
    assert all(set(case.relevant_urls) <= urls for case in cases)
