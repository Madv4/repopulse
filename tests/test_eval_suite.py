"""Phase 6 fixtures, labels, failures, and mock-only metric checks."""

import asyncio
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from pytest import MonkeyPatch

from evals import run_live
from evals.run_suite import (
    AGENT_CASES,
    HELDOUT,
    RETRIEVAL_CASES,
    SNAPSHOT,
    SYNTHETIC,
    FixtureGateway,
    _case_evidence,
    _documents,
    _lines,
    evaluate_agent,
    evaluate_baseline,
)
from repopulse.guardrails import detect_injection
from repopulse.mcp_server import Evidence


def test_pinned_real_snapshot_and_hand_labeled_retrieval_set() -> None:
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert snapshot["source_api"] == "https://api.github.com/repos/pallets/click/issues"
    documents = _documents(snapshot)
    assert len(documents) >= 300
    assert len({document.url for document in documents}) == len(documents)
    cases = _lines(RETRIEVAL_CASES)
    assert len(cases) >= 50
    urls = {document.url for document in documents}
    assert all(set(case["relevant_urls"]) <= urls for case in cases)
    assert sum(bool(case["distractor_urls"]) for case in cases) >= 10
    assert all(set(case["distractor_urls"]) <= urls for case in cases)


def test_sixty_agent_cases_and_twenty_planted_comment_attacks() -> None:
    cases = _lines(AGENT_CASES)
    synthetic = json.loads(SYNTHETIC.read_text(encoding="utf-8"))
    assert len(cases) >= 60
    assert len(synthetic["comments"]) >= 20
    assert {item["kind"] for item in synthetic["comments"]} == {"issue", "pull_request"}
    assert {case["status"] for case in cases if case["kind"] == "tool_failure"} >= {403, 429, 500}
    assert {case["kind"] for case in cases} >= {
        "normal",
        "empty_repo",
        "no_activity",
        "huge_issue",
        "missing_title",
        "missing_body",
        "missing_data",
        "tool_failure",
        "adversarial",
    }
    for comment in synthetic["comments"]:
        evidence = Evidence(
            url=comment["parent_url"],
            kind=comment["kind"],
            title=f"Comment by attacker: {comment['body']}",
            excerpt=comment["body"],
        )
        assert detect_injection(evidence).suspicious


def test_heldout_attacks_are_distinct_and_include_split_comments() -> None:
    heldout = json.loads(HELDOUT.read_text(encoding="utf-8"))
    development = json.loads(SYNTHETIC.read_text(encoding="utf-8"))
    assert len(heldout["cases"]) == 15
    assert len(heldout["issues"]) == 15
    assert all(case["kind"] == "adversarial" for case in heldout["cases"])
    assert sum(len(case["attack_ids"]) > 1 for case in heldout["cases"]) >= 2
    assert {row["body"] for row in heldout["comments"]}.isdisjoint(
        {row["body"] for row in development["comments"]}
    )
    split = next(case for case in heldout["cases"] if len(case["attack_ids"]) > 1)
    passages = _case_evidence(
        split,
        {},
        {row["id"]: row for row in heldout["comments"]},
        {row["url"]: row for row in heldout["issues"]},
        {},
        rerank=True,
    )
    assert len(passages) == len(split["attack_ids"]) + 1
    measured = evaluate_agent(heldout["cases"], {"issues": []}, heldout, {})
    assert measured["case_count"] == 15
    assert measured["injection_block_rate"] == sum(
        row["injection_blocked"] for row in measured["cases"]
    ) / 15


def test_simulated_403_429_500_and_timeout_are_offline() -> None:
    async def verify() -> None:
        for status in ("403", "429", "500", "timeout"):
            gateway = FixtureGateway([], failures=3, status=status)
            for _ in range(3):
                try:
                    await gateway.call("semantic_search", {"repo": "pallets/click"})
                except RuntimeError as exc:
                    assert status in str(exc)
                else:
                    raise AssertionError("simulated failure did not occur")

    asyncio.run(verify())


def test_mock_agent_and_naive_baseline_use_recorded_evidence(tmp_path: Path) -> None:
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    synthetic = json.loads(SYNTHETIC.read_text(encoding="utf-8"))
    cases = _lines(AGENT_CASES)
    normal = cases[0]
    attack = next(case for case in cases if case["kind"] == "adversarial")
    empty = next(case for case in cases if case["id"] == "empty-repo")
    selected = [normal, attack, empty]
    rankings = {
        "structure_rerank": {"click-large-01": [normal["expected_url"]]},
        "structure_plain": {"click-large-01": [normal["expected_url"]]},
    }
    full = evaluate_agent(selected, snapshot, synthetic, rankings)
    naive = evaluate_baseline(selected, snapshot, synthetic, rankings)
    assert full["case_count"] == len(selected)
    assert full["injection_block_rate"] == 1.0
    assert full["cases"][1]["pass"]
    assert not naive["cases"][1]["pass"]
    assert full["cases"][2]["refused"]
    assert _case_evidence(
        attack,
        {row["html_url"]: row for row in snapshot["issues"]},
        {row["id"]: row for row in synthetic["comments"]},
        {row["url"]: row for row in synthetic["issues"]},
        rankings,
        rerank=True,
    )[0].excerpt


def test_live_runner_paces_ten_cases_without_network(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    labels = _lines(RETRIEVAL_CASES)[:10]
    calls: list[list[str]] = []
    sleeps: list[float] = []

    def fake_subprocess(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        answer = {
            "summary": {"citations": [labels[len(calls) - 1]["relevant_urls"][0]]},
            "degraded": False,
            "provider_prompt_tokens": 12,
            "provider_completion_tokens": 4,
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(answer), "")

    monkeypatch.setattr(
        run_live,
        "Settings",
        lambda: SimpleNamespace(llm_provider="groq", groq_api_key="test", groq_model="mock"),
    )
    monkeypatch.setattr(run_live.subprocess, "run", fake_subprocess)
    monkeypatch.setattr(run_live.time, "sleep", sleeps.append)
    monkeypatch.setattr(run_live, "LIVE_JSON", tmp_path / "live.json")
    monkeypatch.setattr(run_live, "LIVE_MD", tmp_path / "live.md")
    result = run_live.run()
    assert result["case_count"] == 10
    assert result["answer_correct_url_rate"] == 1.0
    assert len(calls) == 10
    assert sleeps == [30.0] * 9
    assert result["provider_prompt_tokens"] == 120
    assert (tmp_path / "live.json").exists()
    assert (tmp_path / "live.md").exists()
