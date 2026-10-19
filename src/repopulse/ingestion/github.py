"""Normalize GitHub REST records and ingest a bounded repository date window."""

from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel, Field

from repopulse.github.models import GitHubResponse
from repopulse.ingestion.chunking import Chunker
from repopulse.ingestion.embed import Embedder
from repopulse.ingestion.models import SourceComment, SourceDocument
from repopulse.ingestion.store import ChromaStore


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("GitHub record is missing a timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("GitHub timestamp is missing a timezone")
    return parsed


def _optional_timestamp(value: object) -> datetime | None:
    return _timestamp(value) if value else None


def _author(value: object) -> str:
    return _text(_mapping(value).get("login"))


def _comments(raw: list[dict[str, Any]], start: datetime, end: datetime) -> list[SourceComment]:
    result: list[SourceComment] = []
    for item in raw:
        created = _timestamp(item.get("created_at"))
        if start <= created < end and _text(item.get("body")).strip():
            result.append(
                SourceComment(
                    url=_text(item.get("html_url")),
                    body=_text(item.get("body")),
                    author=_author(item.get("user")),
                    created_at=created,
                )
            )
    return result


def issue_document(
    repo: str,
    raw: dict[str, Any],
    comments: list[SourceComment] | None = None,
) -> SourceDocument:
    """Normalize one REST issue object."""
    number = int(raw["number"])
    labels = raw.get("labels") or []
    return SourceDocument(
        repo=repo,
        kind="issue",
        source_id=f"{repo}:issue:{number}",
        number=number,
        url=_text(raw.get("html_url")),
        title=_text(raw.get("title")),
        body=_text(raw.get("body")),
        author=_author(raw.get("user")),
        created_at=_timestamp(raw.get("created_at")),
        updated_at=_optional_timestamp(raw.get("updated_at")),
        labels=[_text(_mapping(label).get("name")) for label in labels],
        comments=comments or [],
    )


def pull_document(
    repo: str,
    issue: dict[str, Any],
    pull: dict[str, Any],
    comments: list[SourceComment] | None = None,
) -> SourceDocument:
    """Normalize a PR detail plus its issue metadata and comments."""
    number = int(pull["number"])
    labels = issue.get("labels") or pull.get("labels") or []
    return SourceDocument(
        repo=repo,
        kind="pull_request",
        source_id=f"{repo}:pull_request:{number}",
        number=number,
        url=_text(pull.get("html_url")),
        title=_text(pull.get("title")),
        body=_text(pull.get("body")),
        author=_author(pull.get("user")),
        created_at=_timestamp(pull.get("created_at")),
        updated_at=_optional_timestamp(pull.get("updated_at")),
        labels=[_text(_mapping(label).get("name")) for label in labels],
        comments=comments or [],
    )


def commit_document(repo: str, raw: dict[str, Any]) -> SourceDocument:
    """Normalize one REST commit summary, retaining the full message."""
    sha = _text(raw.get("sha"))
    detail = _mapping(raw.get("commit"))
    message = _text(detail.get("message"))
    lines = message.split("\n", 1)
    timestamp = _timestamp(_mapping(detail.get("committer")).get("date"))
    author = _author(raw.get("author")) or _text(_mapping(detail.get("author")).get("name"))
    return SourceDocument(
        repo=repo,
        kind="commit",
        source_id=f"{repo}:commit:{sha}",
        url=_text(raw.get("html_url")),
        title=lines[0],
        body=lines[1] if len(lines) > 1 else "",
        author=author,
        created_at=timestamp,
        updated_at=timestamp,
    )


def _in_window(raw: dict[str, Any], start: datetime, end: datetime) -> bool:
    return any(
        start <= parsed < end
        for key in ("created_at", "updated_at", "closed_at", "merged_at")
        if (parsed := _optional_timestamp(raw.get(key))) is not None
    )


def _utc_string(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class IngestionStats(BaseModel):
    """Actual count of documents and chunks written to Chroma."""

    documents: int = Field(ge=0)
    chunks: int = Field(ge=0)


class GitHubReader(Protocol):
    """Read operations required by the ingestor and recorded fixtures."""

    def repo_path(self, repo: str) -> str: ...

    def list_issues_since(self, repo: str, since: str) -> list[dict[str, Any]]: ...

    def get_pull_request(self, repo: str, number: int) -> GitHubResponse: ...

    def list_issue_comments(self, repo: str, number: int) -> list[dict[str, Any]]: ...

    def list_pr_review_comments(self, repo: str, number: int) -> list[dict[str, Any]]: ...

    def list_commits_between(self, repo: str, since: str, until: str) -> list[dict[str, Any]]: ...


class RepositoryIngestor:
    """Fetch and normalize issues, PRs, comments, and commits for one window."""

    def __init__(self, github: GitHubReader) -> None:
        self.github = github

    def fetch(self, repo: str, start: datetime, end: datetime) -> list[SourceDocument]:
        """Read GitHub records and retain activity in the half-open [start, end) window."""
        self.github.repo_path(repo)
        if start.tzinfo is None or end.tzinfo is None or start >= end:
            raise ValueError("start and end must be timezone-aware with start < end")
        documents: list[SourceDocument] = []
        issues = self.github.list_issues_since(repo, _utc_string(start))
        for issue in issues:
            if not _in_window(issue, start, end):
                continue
            number = int(issue["number"])
            discussion = (
                self.github.list_issue_comments(repo, number)
                if int(issue.get("comments") or 0)
                else []
            )
            if "pull_request" in issue:
                pull_data = self.github.get_pull_request(repo, number).data
                if not isinstance(pull_data, dict):
                    raise ValueError("GitHub pull request response was not an object")
                review = (
                    self.github.list_pr_review_comments(repo, number)
                    if int(pull_data.get("review_comments") or 0)
                    else []
                )
                comments = _comments(discussion + review, start, end)
                documents.append(pull_document(repo, issue, pull_data, comments))
            else:
                documents.append(issue_document(repo, issue, _comments(discussion, start, end)))
        for commit in self.github.list_commits_between(repo, _utc_string(start), _utc_string(end)):
            document = commit_document(repo, commit)
            if start <= document.created_at < end:
                documents.append(document)
        return documents

    def ingest(
        self,
        repo: str,
        start: datetime,
        end: datetime,
        chunker: Chunker,
        store: ChromaStore,
        embedder: Embedder,
    ) -> IngestionStats:
        """Fetch, chunk, embed, and persist a repository window."""
        documents = self.fetch(repo, start, end)
        count = sum(
            store.replace_document(document, chunker.chunk(document), embedder)
            for document in documents
        )
        return IngestionStats(documents=len(documents), chunks=count)
