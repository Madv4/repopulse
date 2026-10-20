"""Offline checks for the recorded benign-comment audit."""

import hashlib
import json

from evals.run_benign_fp import FIXTURE, PARENT, REPORT_JSON, evidence_for, summarize


def test_benign_fixture_is_anchored_to_pinned_parent_snapshot() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    parents = {row["number"]: row for row in parent["issues"]}
    rows = fixture["comments"]
    assert fixture["parent_snapshot_sha256"] == hashlib.sha256(PARENT.read_bytes()).hexdigest()
    assert len(rows) >= 200
    assert len({row["id"] for row in rows}) == len(rows)
    for row in rows:
        assert row["issue_number"] in parents
        assert row["issue_url"] == parents[row["issue_number"]]["html_url"]
        assert row["url"].startswith(row["issue_url"] + "#issuecomment-")
        assert row["body"].strip()
        assert evidence_for(row).excerpt == row["body"][:1800]


def test_quarantine_mode_counts_use_union_and_overlap() -> None:
    rows = [
        {"heuristic_blocked": True, "classifier_blocked": False},
        {"heuristic_blocked": False, "classifier_blocked": True},
        {"heuristic_blocked": True, "classifier_blocked": True},
        {"heuristic_blocked": False, "classifier_blocked": False},
    ]
    assert summarize(rows) == {
        "sample": 4, "heuristic_only_mode": 2, "classifier_only_mode": 2,
        "both_screens_mode": 3, "overlap": 1, "heuristic_exclusive": 1,
        "classifier_exclusive": 1,
    }


def test_committed_live_report_counts_match_saved_case_decisions() -> None:
    report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    assert report["metric_source"] == "heuristic only; live Groq classifier"
    assert report["classifier_model"] == "openai/gpt-oss-20b"
    assert report["comment_fixture_sha256"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert report["counts"] == summarize(report["cases"])
    assert len(report["cases"]) == len(json.loads(FIXTURE.read_text(encoding="utf-8"))["comments"])
