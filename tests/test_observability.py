"""Offline full-run tracing, CLI inspection, and HTML viewer checks."""

import asyncio
import json
import re
import sys
from pathlib import Path

import pytest

from repopulse import cli
from repopulse.agent import RepoPulseAgent
from repopulse.config import Settings
from repopulse.llm.base import ChatMessage
from repopulse.llm.mock import MockProvider
from repopulse.mcp_server import Evidence, ToolEvidence
from repopulse.memory import MemoryStore
from repopulse.observability.tracer import TraceWriter, load_trace
from repopulse.observability.viewer import render_trace_html

SOURCE = Evidence(
    url="https://github.com/pallets/click/issues/7",
    kind="issue",
    title="ANSI color handling report",
    excerpt="The issue describes terminal color output.",
)


class RecordedGateway:
    """Return one recorded passage after an optional number of simulated failures."""

    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.calls = 0

    async def call(self, name: str, arguments: dict[str, str | int]) -> ToolEvidence:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("simulated failure")
        return ToolEvidence(evidence=[SOURCE])


def _reply(messages: tuple[ChatMessage, ...]) -> str:
    instruction = messages[0].content
    if "subquestions" in instruction:
        return '{"subquestions":["find color issue"]}'
    if '"tool"' in instruction:
        return '{"tool":"semantic_search","args":{"query":"color issue"}}'
    if '"action"' in instruction:
        return '{"action":"enough"}'
    return json.dumps(
        {
            "summary": {
                "text": "The issue concerns ANSI color output.",
                "evidence": [{"evidence_id": SOURCE.evidence_id, "quote": SOURCE.title}],
            }
        }
    )


async def _no_sleep(delay: float) -> None:
    return None


def test_complete_trace_records_agent_run_and_memory(tmp_path: Path) -> None:
    trace = TraceWriter(tmp_path / "traces")
    memory = MemoryStore(tmp_path / "memory.sqlite3")
    memory.add_user("I prefer concise issue summaries.", repo="pallets/click")
    agent = RepoPulseAgent(
        MockProvider(responder=_reply),
        RecordedGateway(),
        trace=trace,
        long_term_memory=memory,
        sleep=_no_sleep,
    )
    answer = asyncio.run(agent.ask("Which issue describes ANSI colors?", "pallets/click"))
    events = load_trace(tmp_path / "traces", trace.run_id)
    assert answer.trace_id == trace.run_id
    assert not answer.degraded
    assert events[0]["event"] == "run_start"
    assert events[0]["question"] == "Which issue describes ANSI colors?"
    assert events[-1]["event"] == "run_end"
    kinds = {event["event"] for event in events}
    assert {
        "plan", "llm_call", "tool_call", "evidence", "memory_read", "memory_write",
        "guardrail", "token_usage", "final_answer",
    } <= kinds
    tool = next(event for event in events if event["event"] == "tool_call")
    assert tool["args"]["repo"] == "pallets/click"
    assert tool["result_summary"]["evidence_ids"] == [SOURCE.evidence_id]
    assert tool["latency_ms"] >= 0
    passage = next(event for event in events if event["event"] == "evidence")
    assert passage["excerpt"] == SOURCE.excerpt
    assert passage["disposition"] == "accepted"
    assert any(
        event["event"] == "memory_read" and event["scope"] == "long_term"
        and event["records"][0]["source"] == "user"
        for event in events
    )
    assert any(
        event["event"] == "memory_write" and event["scope"] == "session"
        for event in events
    )
    assert any(
        event["event"] == "memory_write" and event["scope"] == "long_term"
        and event["status"] == "none"
        for event in events
    )
    usage = next(event for event in events if event["event"] == "token_usage")
    assert usage["budget_units_used"] == answer.budget_units_used
    final = next(event for event in events if event["event"] == "final_answer")
    assert final["answer"]["summary"]["citations"] == [SOURCE.url]


def test_trace_records_every_retry_and_result_latency(tmp_path: Path) -> None:
    trace = TraceWriter(tmp_path / "traces")
    agent = RepoPulseAgent(
        MockProvider(responder=_reply), RecordedGateway(failures=2),
        max_steps=3, trace=trace, sleep=_no_sleep,
    )
    asyncio.run(agent.ask("Which issue describes ANSI colors?", "pallets/click"))
    events = load_trace(tmp_path / "traces", trace.run_id)
    calls = [event for event in events if event["event"] == "tool_call"]
    assert [event["attempt"] for event in calls] == [1, 2, 3]
    assert [event["status"] for event in calls] == ["error", "error", "ok"]
    assert [event["result_summary"]["error_type"] for event in calls[:2]] == [
        "RuntimeError", "RuntimeError",
    ]
    assert [event["delay_seconds"] for event in events if event["event"] == "tool_retry"] == [
        1.0, 2.0,
    ]
    assert all(event["latency_ms"] >= 0 for event in calls)


def test_trace_reader_and_viewer_redact_and_escape_local_payload(tmp_path: Path) -> None:
    trace = TraceWriter(tmp_path)
    fake_secret = "gsk_" + "A" * 24
    hostile = "</script><script>alert(1)</script>"
    trace.record("evidence", title=hostile, excerpt=f"token={fake_secret}")
    events = load_trace(tmp_path, trace.run_id)
    assert fake_secret not in json.dumps(events)
    assert "[REDACTED_SECRET]" in events[0]["excerpt"]
    html = render_trace_html(events, trace.run_id, tmp_path / "viewer.html").read_text(
        encoding="utf-8"
    )
    assert hostile not in html
    assert "\\u003c/script\\u003e" in html
    assert "<script src=" not in html
    assert "https://fonts." not in html
    with pytest.raises(ValueError):
        load_trace(tmp_path, "../outside")


def test_trace_show_cli_json_and_html_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    trace = TraceWriter(Path("traces"))
    trace.record("run_start", question="Recorded question", repo="pallets/click")
    monkeypatch.setattr(cli, "Settings", lambda: Settings(_env_file=None, llm_provider="mock"))
    monkeypatch.setattr(sys, "argv", ["repopulse", "trace", "show", trace.run_id])
    cli.main()
    shown = json.loads(capsys.readouterr().out)
    assert shown[0]["question"] == "Recorded question"
    output = tmp_path / "trace-view.html"
    monkeypatch.setattr(
        sys, "argv", ["repopulse", "trace", "show", trace.run_id, "--html", str(output)]
    )
    cli.main()
    assert output.exists()
    assert "Wrote trace viewer" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["repopulse", "trace", "show", "../bad"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


def test_explicit_memory_add_has_a_long_term_write_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings(_env_file=None, llm_provider="mock", memory_db_path=tmp_path / "m.db")
    monkeypatch.setattr(cli, "Settings", lambda: settings)
    monkeypatch.setattr(
        sys, "argv", ["repopulse", "memory", "add", "I prefer concise answers."]
    )
    cli.main()
    output = capsys.readouterr()
    match = re.search(r"Trace: ([0-9a-f]{32})", output.err)
    assert match is not None
    events = load_trace(Path("traces"), match.group(1))
    assert events[0]["event"] == "memory_write"
    assert events[0]["scope"] == "long_term"
    assert events[0]["source"] == "user"
