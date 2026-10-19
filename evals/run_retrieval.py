"""Run four local retrieval variants on a pinned, recorded GitHub snapshot."""

import hashlib
import json
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from evals.metrics import QueryResult, RetrievalCase, aggregate, measure_case
from repopulse.config import Settings
from repopulse.github.client import GitHubClient
from repopulse.github.models import GitHubResponse
from repopulse.ingestion.chunking import FixedChunker, StructureChunker
from repopulse.ingestion.embed import LocalEmbedder
from repopulse.ingestion.github import RepositoryIngestor
from repopulse.ingestion.store import ChromaStore
from repopulse.retrieval.rerank import LocalCrossEncoderReranker
from repopulse.retrieval.retriever import Retriever

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "evals/fixtures/click_snapshot.json"
DATASET = ROOT / "evals/datasets/retrieval.jsonl"
REPORT = ROOT / "evals/reports/retrieval_report.md"
RESULTS = ROOT / "evals/reports/retrieval_results.json"


class VariantResult(BaseModel):
    """Measured metrics and ranked outputs for one retrieval configuration."""

    strategy: str
    rerank: bool
    documents: int
    chunks: int
    hit_rate_at_3: float
    mrr_at_3: float
    cases: list[QueryResult]


class RecordedSnapshot(BaseModel):
    """Validated shape of the committed REST-field snapshot."""

    repo: str
    captured_at: str
    window_start: str
    window_end: str
    issues: list[dict[str, Any]]
    pulls: dict[str, dict[str, Any]]
    issue_comments: dict[str, list[dict[str, Any]]]
    review_comments: dict[str, list[dict[str, Any]]]
    commits: list[dict[str, Any]]


class RecordedGitHubClient:
    """Minimal read-only GitHub client backed entirely by a JSON fixture."""

    def __init__(self, snapshot: RecordedSnapshot) -> None:
        self.snapshot = snapshot

    def repo_path(self, repo: str) -> str:
        """Validate fixture queries against the pinned repository."""
        if repo != self.snapshot.repo:
            raise ValueError("Fixture contains a different repository")
        return GitHubClient.repo_path(repo)

    def list_issues_since(self, repo: str, since: str) -> list[dict[str, Any]]:
        """Return recorded issue and PR summaries."""
        self.repo_path(repo)
        return self.snapshot.issues

    def get_pull_request(self, repo: str, number: int) -> GitHubResponse:
        """Return one recorded PR detail."""
        self.repo_path(repo)
        return GitHubResponse(data=self.snapshot.pulls[str(number)])

    def list_issue_comments(self, repo: str, number: int) -> list[dict[str, Any]]:
        """Return recorded issue discussion comments."""
        self.repo_path(repo)
        return self.snapshot.issue_comments.get(str(number), [])

    def list_issue_comments_since(self, repo: str, since: str) -> list[dict[str, Any]]:
        """Flatten recorded discussion comments with their recorded issue keys."""
        self.repo_path(repo)
        return [
            {**comment, "issue_url": f"https://api.github.com/repos/{repo}/issues/{number}"}
            for number, comments in self.snapshot.issue_comments.items()
            for comment in comments
        ]

    def list_pr_review_comments(self, repo: str, number: int) -> list[dict[str, Any]]:
        """Return recorded line-level PR review comments."""
        self.repo_path(repo)
        return self.snapshot.review_comments.get(str(number), [])

    def list_pr_review_comments_since(self, repo: str, since: str) -> list[dict[str, Any]]:
        """Flatten recorded review comments with their recorded PR keys."""
        self.repo_path(repo)
        return [
            {**comment, "pull_request_url": f"https://api.github.com/repos/{repo}/pulls/{number}"}
            for number, comments in self.snapshot.review_comments.items()
            for comment in comments
        ]

    def list_commits_between(self, repo: str, since: str, until: str) -> list[dict[str, Any]]:
        """Return recorded commit summaries."""
        self.repo_path(repo)
        return self.snapshot.commits


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cases() -> list[RetrievalCase]:
    return [
        RetrievalCase.model_validate_json(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run() -> list[VariantResult]:
    """Measure every strategy/reranker combination without network access."""
    settings = Settings()
    snapshot = RecordedSnapshot.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    cases = _cases()
    start = datetime.fromisoformat(snapshot.window_start.replace("Z", "+00:00"))
    end = datetime.fromisoformat(snapshot.window_end.replace("Z", "+00:00"))
    fixture_client = RecordedGitHubClient(snapshot)
    documents = RepositoryIngestor(fixture_client).fetch(snapshot.repo, start, end)
    available_urls = {document.url for document in documents}
    for case in cases:
        if not set(case.relevant_urls) & available_urls:
            raise ValueError(f"Case {case.id} has no relevant URL in the recorded corpus")
    embedder = LocalEmbedder(
        settings.model_cache_dir, settings.embedding_model, revision=settings.embedding_revision
    )
    reranker = LocalCrossEncoderReranker(
        settings.model_cache_dir, settings.reranker_model, revision=settings.reranker_revision
    )
    variants: list[VariantResult] = []
    for chunker in (FixedChunker(), StructureChunker()):
        directory = settings.chroma_dir.parent / "retrieval-eval"
        store = ChromaStore(directory, chunker.name)
        store.clear_repo(snapshot.repo)
        chunk_count = sum(
            store.replace_document(document, chunker.chunk(document), embedder)
            for document in documents
        )
        retriever = Retriever(store, embedder, reranker)
        for use_reranker in (False, True):
            outcomes = [
                measure_case(
                    case,
                    [
                        hit.chunk.url
                        for hit in retriever.search(
                            case.query, repo=snapshot.repo, k=3, rerank=use_reranker
                        )
                    ],
                    3,
                )
                for case in cases
            ]
            hit_rate, mrr = aggregate(outcomes)
            variants.append(
                VariantResult(
                    strategy=chunker.name,
                    rerank=use_reranker,
                    documents=len(documents),
                    chunks=chunk_count,
                    hit_rate_at_3=hit_rate,
                    mrr_at_3=mrr,
                    cases=outcomes,
                )
            )
    payload = {
        "fixture_sha256": _sha(FIXTURE),
        "dataset_sha256": _sha(DATASET),
        "captured_at": snapshot.captured_at,
        "repo": snapshot.repo,
        "case_count": len(cases),
        "embedding_model": settings.embedding_model,
        "embedding_revision": settings.embedding_revision,
        "reranker_model": settings.reranker_model,
        "reranker_revision": settings.reranker_revision,
        "chromadb_version": version("chromadb"),
        "sentence_transformers_version": version("sentence-transformers"),
        "variants": [variant.model_dump(mode="json") for variant in variants],
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Retrieval evaluation",
        "",
        "Measured by `make eval` on recorded GitHub REST fields; no GitHub or LLM network calls.",
        f"Fixture: `pallets/click`, captured {snapshot.captured_at}; "
        f"SHA-256 `{payload['fixture_sha256']}`.",
        f"Dataset: {len(cases)} labeled queries; SHA-256 `{payload['dataset_sha256']}`.",
        f"Embedding: `{settings.embedding_model}` at `{settings.embedding_revision}`.",
        f"Reranker: `{settings.reranker_model}` at `{settings.reranker_revision}`.",
        f"Packages: ChromaDB {payload['chromadb_version']}, "
        f"sentence-transformers {payload['sentence_transformers_version']}.",
        "",
        "| Chunking | Reranker | Documents | Chunks | Hit rate@3 | MRR@3 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {item.strategy} | {'on' if item.rerank else 'off'} | {item.documents} | "
        f"{item.chunks} | {item.hit_rate_at_3:.3f} | {item.mrr_at_3:.3f} |"
        for item in variants
    )
    lines.extend(
        [
            "",
            "Hit rate@3 is the fraction of queries with a labeled source among the first three "
            "distinct source documents. MRR@3 is mean reciprocal rank of the first labeled "
            "source in those results. Labels were written before running the comparison. Both "
            "chunkers use 500-character limits; fixed chunking overlaps 80 characters. The "
            "cross-encoder reranks up to 20 vector candidates. This small curated dataset is "
            "a retrieval smoke benchmark, not a general accuracy claim.",
            "",
            "Per-query ranked URLs and metrics are in `retrieval_results.json`.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return variants


if __name__ == "__main__":
    for variant in run():
        print(
            f"{variant.strategy} rerank={variant.rerank}: "
            f"hit@3={variant.hit_rate_at_3:.3f}, MRR@3={variant.mrr_at_3:.3f}"
        )
