"""RepoPulse setup, ingestion, retrieval and cited-answer commands."""

import argparse
import asyncio
import os
import sys
from datetime import UTC, date, datetime, time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from repopulse.agent import Answer, RepoPulseAgent
from repopulse.config import Settings
from repopulse.github.client import GitHubClient
from repopulse.github.models import GitHubRateLimitError
from repopulse.ingestion.chunking import FixedChunker, StructureChunker
from repopulse.ingestion.embed import LocalEmbedder
from repopulse.ingestion.github import RepositoryIngestor
from repopulse.ingestion.store import ChromaStore
from repopulse.llm.base import ChatMessage, ProviderError
from repopulse.llm.factory import create_provider
from repopulse.mcp_gateway import MCPGateway
from repopulse.observability.tracer import TraceWriter
from repopulse.retrieval.rerank import LocalCrossEncoderReranker
from repopulse.retrieval.retriever import Retriever


def _midnight(value: str) -> datetime:
    return datetime.combine(date.fromisoformat(value), time.min, tzinfo=UTC)


def _show_answer(answer: Answer, trace: TraceWriter) -> None:
    if answer.degraded:
        reasons = "; ".join(answer.degradation_reasons)
        print(f"DEGRADED: {reasons} Trace: {trace.path}", file=sys.stderr, flush=True)
    print(answer.model_dump_json(indent=2, ensure_ascii=True))


async def _ask(
    question: str, repo: str, settings: Settings, max_steps: int, token_budget: int
) -> None:
    """Run the agent against this installation's local MCP stdio server."""
    print("Loading the cached local embedding model...", file=sys.stderr, flush=True)
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "repopulse.mcp_server"],
        env=os.environ.copy(),
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        print("Planning the answer...", file=sys.stderr, flush=True)
        trace = TraceWriter(Path("traces"))
        agent = RepoPulseAgent(
            create_provider(settings),
            MCPGateway(session),
            max_steps=max_steps,
            token_budget=token_budget,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
            trace=trace,
        )
        answer = await agent.ask(question, repo)
        _show_answer(answer, trace)


def main() -> None:
    """Dispatch local commands without loading an LLM or model unnecessarily."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(prog="repopulse")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Check the selected provider")
    doctor.add_argument("--live", action="store_true", help="Call the selected model once")
    models = commands.add_parser("models", help="Manage free local model weights")
    models.add_argument("action", choices=["prepare"])
    ingest = commands.add_parser("ingest", help="Ingest a GitHub repository date window")
    ingest.add_argument("repo", help="owner/name")
    ingest.add_argument("--since", required=True, help="Inclusive UTC date, YYYY-MM-DD")
    ingest.add_argument("--until", required=True, help="Exclusive UTC date, YYYY-MM-DD")
    ingest.add_argument("--strategy", choices=["fixed", "structure"], default="structure")
    search = commands.add_parser("search", help="Search already-ingested evidence")
    search.add_argument("query")
    search.add_argument("--repo", required=True, help="owner/name")
    search.add_argument("--strategy", choices=["fixed", "structure"], default="structure")
    search.add_argument("--k", type=int, default=5)
    search.add_argument("--rerank", action="store_true")
    ask = commands.add_parser("ask", help="Answer a repository question with cited evidence")
    ask.add_argument("question")
    ask.add_argument("--repo", required=True, help="owner/name")
    ask.add_argument("--max-steps", type=int, default=8)
    ask.add_argument("--token-budget", type=int, default=8000)
    args = parser.parse_args()
    settings = Settings()
    if args.command == "doctor":
        provider = create_provider(settings)
        print(f"RepoPulse ready; LLM provider: {provider.name}")
        print(
            "GitHub authentication: configured"
            if settings.github_token
            else "GitHub: public access"
        )
        if args.live:
            try:
                provider.complete(
                    [ChatMessage(role="user", content="Reply with OK.")],
                    max_output_tokens=128,
                )
            except ProviderError as exc:
                parser.exit(2, f"Live provider check failed: {exc}\n")
            model = {
                "ollama": settings.ollama_model,
                "groq": settings.groq_model,
                "gemini": settings.gemini_model,
                "mock": "mock",
            }[provider.name]
            print(f"Live provider check passed; model accepted: {model}")
    elif args.command == "models":
        LocalEmbedder(
            settings.model_cache_dir,
            settings.embedding_model,
            revision=settings.embedding_revision,
            local_files_only=False,
        ).load()
        LocalCrossEncoderReranker(
            settings.model_cache_dir,
            settings.reranker_model,
            revision=settings.reranker_revision,
            local_files_only=False,
        ).load()
        print(f"Local embedding and reranker models cached in {settings.model_cache_dir}")
    elif args.command == "ingest":
        chunker = FixedChunker() if args.strategy == "fixed" else StructureChunker()
        try:
            stats = RepositoryIngestor(GitHubClient(settings)).ingest(
                args.repo,
                _midnight(args.since),
                _midnight(args.until),
                chunker,
                ChromaStore(settings.chroma_dir, chunker.name),
                LocalEmbedder(
                    settings.model_cache_dir,
                    settings.embedding_model,
                    revision=settings.embedding_revision,
                ),
            )
        except GitHubRateLimitError as exc:
            hint = (
                "A free GitHub personal access token in local .env as GITHUB_TOKEN "
                "can raise the limit; otherwise retry after the reset."
                if not settings.github_token
                else "Retry after the GitHub reset."
            )
            parser.exit(2, f"{exc}. {hint}\n")
        print(f"Ingested {stats.documents} documents and {stats.chunks} chunks")
    elif args.command == "search":
        store = ChromaStore(settings.chroma_dir, args.strategy)
        reranker = (
            LocalCrossEncoderReranker(
                settings.model_cache_dir,
                settings.reranker_model,
                revision=settings.reranker_revision,
            )
            if args.rerank
            else None
        )
        retriever = Retriever(
            store,
            LocalEmbedder(
                settings.model_cache_dir,
                settings.embedding_model,
                revision=settings.embedding_revision,
            ),
            reranker,
        )
        for rank, hit in enumerate(
            retriever.search(args.query, repo=args.repo, k=args.k, rerank=args.rerank), start=1
        ):
            print(f"{rank}. {hit.chunk.url}  distance={hit.distance:.4f}")
            print(hit.chunk.text[:300].replace("\n", " "))
    elif args.command == "ask":
        asyncio.run(_ask(args.question, args.repo, settings, args.max_steps, args.token_budget))


if __name__ == "__main__":
    main()
