"""MCP tool registration and recorded GitHub field tests, with no network."""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from repopulse.github.models import GitHubResponse
from repopulse.mcp_gateway import MCPGateway
from repopulse.mcp_server import ToolService, create_server
from repopulse.retrieval.models import SearchHit

FIXTURE = Path(__file__).resolve().parents[1] / "evals/fixtures/click_snapshot.json"
SNAPSHOT = json.loads(FIXTURE.read_text(encoding="utf-8"))


class RecordedReader:
    """Serve the captured public repository fields to every MCP GitHub tool."""

    def search_issues(self, repo: str, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        assert repo == SNAPSHOT["repo"]
        words = query.lower().split()
        return [
            row
            for row in SNAPSHOT["issues"]
            if "pull_request" not in row
            and any(word in (row["title"] + " " + row["body"]).lower() for word in words)
        ][:limit]

    def get_issue(self, repo: str, number: int) -> GitHubResponse:
        assert repo == SNAPSHOT["repo"]
        return GitHubResponse(
            data=next(row for row in SNAPSHOT["issues"] if row["number"] == number)
        )

    def get_pull_request(self, repo: str, number: int) -> GitHubResponse:
        assert repo == SNAPSHOT["repo"]
        return GitHubResponse(data=SNAPSHOT["pulls"][str(number)])

    def get_pr_files(self, repo: str, number: int, *, limit: int = 10) -> list[dict[str, Any]]:
        assert repo == SNAPSHOT["repo"] and str(number) in SNAPSHOT["pulls"]
        return []  # The Phase 2 snapshot did not record changed-file lists.

    def list_commits(self, repo: str, *, limit: int = 5) -> list[dict[str, Any]]:
        assert repo == SNAPSHOT["repo"]
        return SNAPSHOT["commits"][:limit]

    def list_milestones(self, repo: str, *, limit: int = 5) -> list[dict[str, Any]]:
        assert repo == SNAPSHOT["repo"]
        return []  # The Phase 2 snapshot did not record milestones.

    def list_releases(self, repo: str, *, limit: int = 5) -> list[dict[str, Any]]:
        assert repo == SNAPSHOT["repo"]
        return []  # The Phase 2 snapshot did not record releases.


class EmptySearcher:
    def search(self, query: str, *, repo: str, k: int = 5, rerank: bool = False) -> list[SearchHit]:
        assert repo == SNAPSHOT["repo"] and query and k > 0 and not rerank
        return []


def test_all_eight_mcp_tools_have_typed_schemas_and_descriptions() -> None:
    server = create_server(ToolService(RecordedReader(), EmptySearcher))
    tools = asyncio.run(server.list_tools())
    assert {tool.name for tool in tools} == {
        "search_issues",
        "get_issue",
        "get_pull_request",
        "get_pr_files",
        "list_commits",
        "list_milestones",
        "list_releases",
        "semantic_search",
    }
    assert all(tool.description and "repo" in tool.inputSchema["properties"] for tool in tools)


def test_tools_preserve_recorded_source_urls() -> None:
    service = ToolService(RecordedReader(), EmptySearcher)
    repo = SNAPSHOT["repo"]
    issue = next(row for row in SNAPSHOT["issues"] if "pull_request" not in row)
    result = service.get_issue(repo, issue["number"])
    assert result.evidence[0].url == issue["html_url"]
    assert result.evidence[0].title == issue["title"]
    assert any(e.url == issue["html_url"] for e in service.search_issues(repo, "ANSI").evidence)
    commit = service.list_commits(repo, limit=1).evidence[0]
    assert commit.url == SNAPSHOT["commits"][0]["html_url"]
    assert service.semantic_search(repo, "ANSI").evidence == []


def test_pr_and_unrecorded_tool_fields_are_explicitly_empty() -> None:
    service = ToolService(RecordedReader(), EmptySearcher)
    repo = SNAPSHOT["repo"]
    number = int(next(iter(SNAPSHOT["pulls"])))
    assert service.get_pull_request(repo, number).evidence[0].url.endswith(f"/pull/{number}")
    assert service.get_pr_files(repo, number).evidence == []
    assert service.list_milestones(repo).evidence == []
    assert service.list_releases(repo).evidence == []


def test_stdio_mcp_round_trip_uses_recorded_fixture() -> None:
    async def round_trip() -> None:
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(Path(__file__).with_name("fixture_mcp_server.py"))],
            env=os.environ.copy(),
        )
        async with (
            stdio_client(params) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            result = await MCPGateway(session).call(
                "search_issues", {"repo": SNAPSHOT["repo"], "query": "ANSI"}
            )
            assert result.evidence
            assert all(
                item.url.startswith("https://github.com/pallets/click/")
                for item in result.evidence
            )

    asyncio.run(round_trip())
