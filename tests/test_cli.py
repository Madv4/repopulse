"""CLI failure handling without external GitHub or LLM calls."""

import json
import sys
from pathlib import Path

import pytest

from repopulse import cli
from repopulse.agent.graph import Answer
from repopulse.config import Settings
from repopulse.github.models import GitHubRateLimitError
from repopulse.llm.base import ProviderError
from repopulse.llm.mock import MockProvider
from repopulse.memory import MemoryStore
from repopulse.observability.tracer import TraceWriter


def test_ingest_rate_limit_shows_actionable_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    class RateLimitedIngestor:
        def __init__(self, github: object) -> None:
            pass

        def ingest(self, *args: object) -> None:
            raise GitHubRateLimitError(120)

    monkeypatch.setattr(cli, "RepositoryIngestor", RateLimitedIngestor)
    monkeypatch.setattr(
        cli,
        "Settings",
        lambda: Settings(
            _env_file=None,
            github_cache_dir=tmp_path / "cache",
            chroma_dir=tmp_path / "chroma",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["repopulse", "ingest", "a/b", "--since", "2026-09-01", "--until", "2026-10-01"],
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "retry after about 120 seconds" in error
    assert "GITHUB_TOKEN" in error
    assert "Traceback" not in error


def test_degraded_answer_has_visible_cli_line(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    trace = TraceWriter(tmp_path / "traces")
    trace.degradation("Synthesis LLM request failed: HTTP 404.")
    answer = Answer(
        degraded=True,
        degradation_reasons=["Synthesis LLM request failed: HTTP 404."],
        trace_id=trace.run_id,
    )
    cli._show_answer(answer, trace)
    output = capsys.readouterr()
    assert "DEGRADED:" in output.err
    assert str(trace.path) in output.err
    assert json.loads(output.out)["degraded"] is True


def test_doctor_live_calls_selected_provider_offline(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = MockProvider(responses=["OK"])
    monkeypatch.setattr(cli, "create_provider", lambda settings: provider)
    monkeypatch.setattr(cli, "Settings", lambda: Settings(_env_file=None, llm_provider="mock"))
    monkeypatch.setattr(sys, "argv", ["repopulse", "doctor", "--live"])
    cli.main()
    assert len(provider.calls) == 1
    assert "model accepted: mock" in capsys.readouterr().out


def test_doctor_live_reports_rejected_model_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def rejected(_messages: object) -> str:
        raise ProviderError("LLM service returned HTTP 404")

    provider = MockProvider(responder=rejected)
    monkeypatch.setattr(cli, "create_provider", lambda settings: provider)
    monkeypatch.setattr(cli, "Settings", lambda: Settings(_env_file=None, llm_provider="mock"))
    monkeypatch.setattr(sys, "argv", ["repopulse", "doctor", "--live"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "HTTP 404" in error
    assert "Traceback" not in error


def test_memory_cli_add_list_and_forget_offline(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    settings = Settings(
        _env_file=None, llm_provider="mock", memory_db_path=tmp_path / "memory.sqlite3"
    )
    monkeypatch.setattr(cli, "Settings", lambda: settings)
    monkeypatch.setattr(
        sys,
        "argv",
        ["repopulse", "memory", "add", "I care about auth changes.", "--repo", "pallets/click"],
    )
    cli.main()
    added = json.loads(capsys.readouterr().out)
    assert added["source"] == "user"
    monkeypatch.setattr(sys, "argv", ["repopulse", "memory", "list", "--repo", "pallets/click"])
    cli.main()
    listed = json.loads(capsys.readouterr().out)
    assert [row["id"] for row in listed] == [added["id"]]
    monkeypatch.setattr(sys, "argv", ["repopulse", "memory", "forget", added["id"]])
    cli.main()
    assert "Forgot memory" in capsys.readouterr().out
    assert MemoryStore(settings.memory_db_path).list_active() == []
