"""CLI failure handling without external GitHub or LLM calls."""

import sys
from pathlib import Path

import pytest

from repopulse import cli
from repopulse.config import Settings
from repopulse.github.models import GitHubRateLimitError


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
