"""Offline ingestion, chunking, Chroma persistence, and retrieval tests."""

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from repopulse.config import Settings
from repopulse.github.client import GitHubClient
from repopulse.ingestion.chunking import FixedChunker, StructureChunker
from repopulse.ingestion.github import RepositoryIngestor
from repopulse.ingestion.models import SourceComment, SourceDocument
from repopulse.ingestion.store import ChromaStore
from repopulse.retrieval.retriever import Retriever


class KeywordEmbedder:
    """Deterministic test vectors with no model loading or network."""

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Represent two marker terms in a tiny fixed space."""
        return [
            [float("alpha" in text.lower()), float("beta" in text.lower()), 0.1] for text in texts
        ]


class ReverseReranker:
    """Give beta passages higher scores to exercise reranking."""

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Return deterministic passage scores."""
        return [1.0 if "beta" in passage.lower() else 0.0 for passage in passages]


def document(number: int, text: str, *, repo: str = "a/b") -> SourceDocument:
    """Build a small source for chunking and storage tests."""
    return SourceDocument(
        repo=repo,
        kind="issue",
        source_id=f"{repo}:issue:{number}",
        number=number,
        url=f"https://github.com/{repo}/issues/{number}",
        title=text,
        body=f"Details about {text}.",
        author="tester",
        created_at=datetime(2026, 9, 2, tzinfo=UTC),
        labels=["bug"],
        comments=[
            SourceComment(
                url=f"https://github.com/{repo}/issues/{number}#issuecomment-1",
                body="A useful follow-up",
                author="reviewer",
                created_at=datetime(2026, 9, 3, tzinfo=UTC),
            )
        ],
    )


def test_chunkers_keep_evidence_and_bound_sizes() -> None:
    source = document(1, "alpha")
    source.body = "# Heading\n\n" + "long passage " * 50
    fixed = FixedChunker(size=100, overlap=20).chunk(source)
    structured = StructureChunker(max_chars=100).chunk(source)
    assert len(fixed) > 1 and len(structured) > 1
    assert all(len(chunk.text) <= 100 and chunk.url == source.url for chunk in fixed + structured)
    assert any("Comment by reviewer" in chunk.text for chunk in structured)
    assert fixed[0].strategy == "fixed" and structured[0].strategy == "structure"
    assert FixedChunker(size=100, overlap=20).chunk(source)[0].id == fixed[0].id
    with pytest.raises(ValueError):
        FixedChunker(size=20, overlap=20)


def test_persistent_chroma_replacement_and_reranking(tmp_path: Path) -> None:
    embedder = KeywordEmbedder()
    store = ChromaStore(tmp_path, "structure")
    chunker = StructureChunker()
    alpha = document(1, "alpha")
    beta = document(2, "beta")
    other = document(3, "alpha", repo="other/repo")
    for source in (alpha, beta, other):
        store.replace_document(source, chunker.chunk(source), embedder)
    reopened = ChromaStore(tmp_path, "structure")
    retriever = Retriever(reopened, embedder, ReverseReranker())
    plain = retriever.search("alpha", repo="a/b", k=2)
    reranked = retriever.search("alpha", repo="a/b", k=2, rerank=True)
    assert [hit.chunk.number for hit in plain] == [1, 2]
    assert [hit.chunk.number for hit in reranked] == [2, 1]
    assert all(hit.chunk.repo == "a/b" for hit in plain)
    alpha.body = "Updated details"
    reopened.replace_document(alpha, chunker.chunk(alpha), embedder)
    assert reopened.collection.get(where={"source_id": alpha.source_id})["ids"] == [
        chunker.chunk(alpha)[0].id
    ]
    old_ids = reopened.collection.get(where={"source_id": alpha.source_id})["ids"]

    class BrokenEmbedder:
        def encode(self, texts: Sequence[str]) -> list[list[float]]:
            return []

    with pytest.raises(ValueError, match="wrong number"):
        reopened.replace_document(alpha, chunker.chunk(alpha), BrokenEmbedder())
    assert reopened.collection.get(where={"source_id": alpha.source_id})["ids"] == old_ids
    reopened.clear_repo("a/b")
    assert retriever.search("alpha", repo="a/b") == []
    assert retriever.search("alpha", repo="other/repo")


def test_github_window_ingests_issue_pr_comments_and_commit(tmp_path: Path) -> None:
    start = datetime(2026, 9, 1, tzinfo=UTC)
    end = datetime(2026, 10, 1, tzinfo=UTC)

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/issues"):
            assert request.url.params["since"] == "2026-09-01T00:00:00Z"
            return httpx.Response(
                200,
                json=[
                    {
                        "number": 1,
                        "html_url": "https://github.com/a/b/issues/1",
                        "title": "alpha bug",
                        "body": "alpha detail",
                        "user": {"login": "dev"},
                        "created_at": "2026-09-03T00:00:00Z",
                        "updated_at": "2026-09-04T00:00:00Z",
                        "labels": [{"name": "bug"}],
                        "comments": 1,
                    },
                    {
                        "number": 2,
                        "html_url": "https://github.com/a/b/pull/2",
                        "title": "beta fix",
                        "body": "",
                        "user": {"login": "dev"},
                        "created_at": "2026-09-05T00:00:00Z",
                        "updated_at": "2026-09-06T00:00:00Z",
                        "labels": [],
                        "comments": 0,
                        "pull_request": {"url": "example"},
                    },
                    {
                        "number": 3,
                        "html_url": "https://github.com/a/b/issues/3",
                        "title": "out of range",
                        "body": "",
                        "user": {"login": "dev"},
                        "created_at": "2026-10-04T00:00:00Z",
                        "updated_at": "2026-10-04T00:00:00Z",
                        "labels": [],
                        "comments": 0,
                    },
                ],
            )
        if path.endswith("/issues/1/comments"):
            return httpx.Response(
                200,
                json=[
                    {
                        "html_url": "https://github.com/a/b/issues/1#issuecomment-7",
                        "body": "follow-up alpha",
                        "user": {"login": "peer"},
                        "created_at": "2026-09-07T00:00:00Z",
                    }
                ],
            )
        if path.endswith("/pulls/2"):
            return httpx.Response(
                200,
                json={
                    "number": 2,
                    "html_url": "https://github.com/a/b/pull/2",
                    "title": "beta fix",
                    "body": "beta detail",
                    "user": {"login": "dev"},
                    "created_at": "2026-09-05T00:00:00Z",
                    "updated_at": "2026-09-06T00:00:00Z",
                    "review_comments": 1,
                },
            )
        if path.endswith("/pulls/2/comments"):
            return httpx.Response(
                200,
                json=[
                    {
                        "html_url": "https://github.com/a/b/pull/2#discussion-r2",
                        "body": "review beta",
                        "user": {"login": "reviewer"},
                        "created_at": "2026-09-08T00:00:00Z",
                    }
                ],
            )
        if path.endswith("/commits"):
            return httpx.Response(
                200,
                json=[
                    {
                        "sha": "abc123",
                        "html_url": "https://github.com/a/b/commit/abc123",
                        "author": {"login": "dev"},
                        "commit": {
                            "message": "gamma change\n\nDetails",
                            "committer": {"date": "2026-09-09T00:00:00Z"},
                        },
                    }
                ],
            )
        raise AssertionError(path)

    github = GitHubClient(
        Settings(_env_file=None, github_cache_dir=tmp_path / "cache"),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    ingestor = RepositoryIngestor(github)
    sources = ingestor.fetch("a/b", start, end)
    assert [source.kind for source in sources] == ["issue", "pull_request", "commit"]
    assert sources[0].labels == ["bug"]
    assert sources[0].comments[0].body == "follow-up alpha"
    assert sources[1].comments[0].body == "review beta"
    assert sources[2].title == "gamma change"
    store = ChromaStore(tmp_path / "chroma", "fixed")
    stats = ingestor.ingest("a/b", start, end, FixedChunker(), store, KeywordEmbedder())
    assert stats.documents == 3 and stats.chunks >= 3
