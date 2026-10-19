"""Read-only MCP tools for GitHub and locally indexed repository evidence."""

from collections.abc import Callable
from typing import Any, Protocol

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from repopulse.config import Settings
from repopulse.github.client import GitHubClient
from repopulse.github.models import GitHubResponse
from repopulse.ingestion.embed import LocalEmbedder
from repopulse.ingestion.store import ChromaStore
from repopulse.retrieval.retriever import Retriever


class GitHubReader(Protocol):
    """Subset of read-only GitHub operations needed by MCP."""

    def search_issues(self, repo: str, query: str, *, limit: int = 5) -> list[dict[str, Any]]: ...
    def get_issue(self, repo: str, number: int) -> GitHubResponse: ...
    def get_pull_request(self, repo: str, number: int) -> GitHubResponse: ...
    def get_pr_files(self, repo: str, number: int, *, limit: int = 10) -> list[dict[str, Any]]: ...
    def list_commits(self, repo: str, *, limit: int = 5) -> list[dict[str, Any]]: ...
    def list_milestones(self, repo: str, *, limit: int = 5) -> list[dict[str, Any]]: ...
    def list_releases(self, repo: str, *, limit: int = 5) -> list[dict[str, Any]]: ...


class Searcher(Protocol):
    """Local search contract; injected with fake hits in offline tests."""

    def search(self, query: str, *, repo: str, k: int = 5, rerank: bool = False) -> Any: ...


class Evidence(BaseModel):
    """Bounded text with an issue, PR or commit citation URL."""

    url: str
    kind: str
    title: str
    excerpt: str = Field(max_length=1800)


class ToolEvidence(BaseModel):
    """A tool result whose citation keys refer to retrieved source records."""

    evidence: list[Evidence] = Field(default_factory=list)


def _text(value: object, limit: int = 1800) -> str:
    return str(value or "")[:limit]


def _record(row: dict[str, Any], kind: str, *, repo: str) -> Evidence:
    url = row.get("html_url")
    path = {"issue": "issues", "pull_request": "pull", "commit": "commit"}[kind]
    if not isinstance(url, str) or not url.startswith(f"https://github.com/{repo}/{path}/"):
        raise ValueError("GitHub record has no in-repository citation URL")
    commit = row.get("commit")
    message = commit.get("message", "") if isinstance(commit, dict) else ""
    first_line = message.splitlines()[0] if message.splitlines() else ""
    title = _text(row.get("title") or first_line, 300)
    return Evidence(
        url=url,
        kind=kind,
        title=title,
        excerpt=_text(row.get("body") or message),
    )


class ToolService:
    """Bounded, source-preserving tool implementation shared by MCP handlers."""

    def __init__(self, github: GitHubReader, searcher: Callable[[], Searcher]) -> None:
        self.github = github
        self._searcher = searcher

    @staticmethod
    def _repo(repo: str) -> str:
        GitHubClient.repo_path(repo)
        return repo

    def search_issues(self, repo: str, query: str, limit: int = 5) -> ToolEvidence:
        """Search issue titles and bodies using GitHub REST search."""
        repo = self._repo(repo)
        return ToolEvidence(
            evidence=[
                _record(row, "issue", repo=repo)
                for row in self.github.search_issues(repo, query, limit=min(max(limit, 1), 10))
            ]
        )

    def get_issue(self, repo: str, number: int) -> ToolEvidence:
        """Fetch one issue's current title and body."""
        repo = self._repo(repo)
        row = self.github.get_issue(repo, number).data
        return (
            ToolEvidence(evidence=[_record(row, "issue", repo=repo)])
            if isinstance(row, dict)
            else ToolEvidence()
        )

    def get_pull_request(self, repo: str, number: int) -> ToolEvidence:
        """Fetch one pull request's current title and body."""
        repo = self._repo(repo)
        row = self.github.get_pull_request(repo, number).data
        return (
            ToolEvidence(evidence=[_record(row, "pull_request", repo=repo)])
            if isinstance(row, dict)
            else ToolEvidence()
        )

    def get_pr_files(self, repo: str, number: int) -> ToolEvidence:
        """Fetch changed filenames and status for one pull request."""
        repo = self._repo(repo)
        url = f"https://github.com/{repo}/pull/{number}"
        return ToolEvidence(
            evidence=[
                Evidence(
                    url=url,
                    kind="pull_request",
                    title=_text(row.get("filename"), 300),
                    excerpt=_text(f"{row.get('status', '')}: {row.get('patch', '')}"),
                )
                for row in self.github.get_pr_files(repo, number, limit=10)
            ]
        )

    def list_commits(self, repo: str, limit: int = 5) -> ToolEvidence:
        """List recent commits with their canonical citation URLs."""
        repo = self._repo(repo)
        return ToolEvidence(
            evidence=[
                _record(row, "commit", repo=repo)
                for row in self.github.list_commits(repo, limit=min(max(limit, 1), 10))
            ]
        )

    def list_milestones(self, repo: str, limit: int = 5) -> ToolEvidence:
        """List milestones as context; milestone URLs are not answer citations."""
        repo = self._repo(repo)
        return ToolEvidence(
            evidence=[
                Evidence(
                    url=str(row.get("html_url") or f"https://github.com/{repo}/milestones"),
                    kind="milestone",
                    title=_text(row.get("title"), 300),
                    excerpt=_text(row.get("description")),
                )
                for row in self.github.list_milestones(repo, limit=min(max(limit, 1), 10))
            ]
        )

    def list_releases(self, repo: str, limit: int = 5) -> ToolEvidence:
        """List releases as context; release URLs are not answer citations."""
        repo = self._repo(repo)
        return ToolEvidence(
            evidence=[
                Evidence(
                    url=str(row.get("html_url") or f"https://github.com/{repo}/releases"),
                    kind="release",
                    title=_text(row.get("name") or row.get("tag_name"), 300),
                    excerpt=_text(row.get("body")),
                )
                for row in self.github.list_releases(repo, limit=min(max(limit, 1), 10))
            ]
        )

    def semantic_search(self, repo: str, query: str, limit: int = 5) -> ToolEvidence:
        """Search locally ingested issue, PR and commit passages."""
        repo = self._repo(repo)
        hits = self._searcher().search(query, repo=repo, k=min(max(limit, 1), 10))
        return ToolEvidence(
            evidence=[
                Evidence(
                    url=hit.chunk.url,
                    kind=hit.chunk.kind,
                    title=_text(
                        hit.chunk.text.partition("\n")[0].strip() or hit.chunk.source_id, 300
                    ),
                    excerpt=_text(hit.chunk.text),
                )
                for hit in hits
            ]
        )


def create_server(service: ToolService) -> FastMCP:
    """Register typed, documented handlers on a local MCP stdio server."""
    server = FastMCP("RepoPulse")

    @server.tool(description="Search GitHub issues in owner/name; return citable issue evidence.")
    def search_issues(repo: str, query: str, limit: int = 5) -> ToolEvidence:
        return service.search_issues(repo, query, limit)

    @server.tool(description="Get one GitHub issue by number and return its current body.")
    def get_issue(repo: str, number: int) -> ToolEvidence:
        return service.get_issue(repo, number)

    @server.tool(description="Get one pull request by number and return its current body.")
    def get_pull_request(repo: str, number: int) -> ToolEvidence:
        return service.get_pull_request(repo, number)

    @server.tool(description="Get changed files for a pull request, cited to the PR URL.")
    def get_pr_files(repo: str, number: int) -> ToolEvidence:
        return service.get_pr_files(repo, number)

    @server.tool(description="List recent commits with messages and commit URLs.")
    def list_commits(repo: str, limit: int = 5) -> ToolEvidence:
        return service.list_commits(repo, limit)

    @server.tool(description="List repository milestones for planning context.")
    def list_milestones(repo: str, limit: int = 5) -> ToolEvidence:
        return service.list_milestones(repo, limit)

    @server.tool(description="List repository releases for planning context.")
    def list_releases(repo: str, limit: int = 5) -> ToolEvidence:
        return service.list_releases(repo, limit)

    @server.tool(description="Search locally indexed repository passages using cached embeddings.")
    def semantic_search(repo: str, query: str, limit: int = 5) -> ToolEvidence:
        return service.semantic_search(repo, query, limit)

    return server


def main() -> None:
    """Start the standalone MCP server over stdio."""
    settings = Settings()
    embedder = LocalEmbedder(
        settings.model_cache_dir,
        settings.embedding_model,
        revision=settings.embedding_revision,
    )
    embedder.load()

    def local_searcher() -> Retriever:
        return Retriever(
            ChromaStore(settings.chroma_dir, "structure"),
            embedder,
        )

    create_server(ToolService(GitHubClient(settings), local_searcher)).run(transport="stdio")


if __name__ == "__main__":
    main()
