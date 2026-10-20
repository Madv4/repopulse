"""The portfolio demo uses recorded evidence and writes an inspectable trace."""

from pathlib import Path

from evals.run_demo import run_demo
from repopulse.observability.tracer import load_trace


def test_demo_catches_injected_comment_and_writes_trace(tmp_path: Path) -> None:
    result = run_demo(tmp_path / "traces")
    events = load_trace(result.trace_path.parent, result.trace_path.stem)
    assert result.summary == "Synthetic repository report 1"
    assert result.citations == ["https://github.com/repopulse/synthetic/issues/1"]
    assert not result.degraded
    assert result.quarantined == 1
    assert any(event["event"] == "quarantine" for event in events)
    assert any(event["event"] == "final_answer" for event in events)
    assert result.viewer_path.is_file()
    assert "<html" in result.viewer_path.read_text(encoding="utf-8").lower()
