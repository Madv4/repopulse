"""Offline checks that badges reflect saved evaluation data."""

import json
from pathlib import Path

import pytest

from evals.generate_badges import generate


def test_eval_badge_reads_report_rate_and_detects_staleness(tmp_path: Path) -> None:
    report = tmp_path / "latest.json"
    report.write_text(json.dumps({
        "metric_source": "mock LLM",
        "full_agent": {"pass_rate": 2 / 3},
        "config": {"min_full_agent_pass_rate": 0.5},
        "ci_pass": True,
    }), encoding="utf-8")
    badges = tmp_path / "badges"
    generate(report, badges, "pending")
    assert "66.7% pass" in (badges / "eval.svg").read_text(encoding="utf-8")
    assert "awaiting first push" in (badges / "ci.svg").read_text(encoding="utf-8")
    generate(report, badges, "passing", check_eval=True)
    assert "passing" in (badges / "ci.svg").read_text(encoding="utf-8")
    report.write_text(json.dumps({
        "metric_source": "mock LLM",
        "full_agent": {"pass_rate": 0.4},
        "config": {"min_full_agent_pass_rate": 0.5},
        "ci_pass": False,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        generate(report, badges, "passing", check_eval=True)
