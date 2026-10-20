"""RepoPulse setup, ingestion, retrieval and cited-answer commands."""

import argparse
import asyncio
import json
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
from repopulse.guardrails import Guardrails
from repopulse.ingestion.chunking import FixedChunker, StructureChunker
from repopulse.ingestion.embed import LocalEmbedder
from repopulse.ingestion.github import RepositoryIngestor
from repopulse.ingestion.store import ChromaStore
from repopulse.llm.base import ChatMessage, ProviderError
from repopulse.llm.factory import create_provider
from repopulse.mcp_gateway import MCPGateway
from repopulse.memory import MemoryStore, SessionMemory
from repopulse.observability.tracer import TraceWriter, load_trace
from repopulse.observability.viewer import render_trace_html
from repopulse.retrieval.rerank import LocalCrossEncoderReranker
from repopulse.retrieval.retriever import Retriever


def _midnight(value: str) -> datetime:
    return datetime.combine(date.fromisoformat(value), time.min, tzinfo=UTC)


def _end_of_day(value: str) -> datetime:
    return datetime.combine(date.fromisoformat(value), time.max, tzinfo=UTC)


def _show_answer(answer: Answer, trace: TraceWriter) -> None:
    if answer.degraded:
        reasons = "; ".join(answer.degradation_reasons)
        print(f"DEGRADED: {reasons} Trace: {trace.path}", file=sys.stderr, flush=True)
    print(answer.model_dump_json(indent=2, ensure_ascii=True))


def _guardrails(settings: Settings) -> Guardrails:
    return Guardrails(
        input_validation=settings.guard_input_validation,
        injection_detection=settings.guard_injection_detection,
        untrusted_wrapping=settings.guard_untrusted_wrapping,
        output_redaction=settings.guard_output_redaction,
        structured_validation=settings.guard_structured_validation,
        citation_verification=settings.guard_citation_verification,
        action_gate=settings.guard_action_gate,
    )


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
            long_term_memory=MemoryStore(settings.memory_db_path),
            guardrails=_guardrails(settings),
        )
        answer = await agent.ask(question, repo)
        _show_answer(answer, trace)


async def _chat(repo: str, settings: Settings, max_steps: int, token_budget: int) -> None:
    """Keep short-term conversation state alive for an interactive CLI session."""
    print("Loading the cached local embedding model...", file=sys.stderr, flush=True)
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "repopulse.mcp_server"], env=os.environ.copy()
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as mcp_session:
        await mcp_session.initialize()
        session_memory = SessionMemory()
        long_term_memory = MemoryStore(settings.memory_db_path)
        provider = create_provider(settings)
        print("Enter questions; type exit to finish.", file=sys.stderr, flush=True)
        while True:
            try:
                question = await asyncio.to_thread(input, "repopulse> ")
            except (EOFError, KeyboardInterrupt):
                break
            if question.strip().lower() in {"exit", "quit"}:
                break
            if not question.strip():
                continue
            trace = TraceWriter(Path("traces"))
            agent = RepoPulseAgent(
                provider,
                MCPGateway(mcp_session),
                max_steps=max_steps,
                token_budget=token_budget,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
                trace=trace,
                session_memory=session_memory,
                long_term_memory=long_term_memory,
                guardrails=_guardrails(settings),
            )
            _show_answer(await agent.ask(question, repo), trace)


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
    chat = commands.add_parser("chat", help="Ask multiple questions in one memory session")
    chat.add_argument("--repo", required=True, help="owner/name")
    chat.add_argument("--max-steps", type=int, default=8)
    chat.add_argument("--token-budget", type=int, default=8000)
    memory = commands.add_parser("memory", help="Manage durable SQLite memory")
    memory_actions = memory.add_subparsers(dest="memory_action", required=True)
    memory_list = memory_actions.add_parser("list", help="List active memories")
    memory_list.add_argument("--repo", help="Include global and this repo's memories")
    memory_add = memory_actions.add_parser("add", help="Save an explicit user fact or preference")
    memory_add.add_argument("text")
    memory_add.add_argument("--repo", help="Scope to owner/name")
    memory_add.add_argument("--kind", choices=["preference", "fact"], default="preference")
    memory_add.add_argument("--confidence", type=float, default=1.0)
    memory_add.add_argument("--expires", help="Last active UTC day, YYYY-MM-DD")
    memory_forget = memory_actions.add_parser("forget", help="Delete a memory by ID")
    memory_forget.add_argument("id")
    trace_command = commands.add_parser("trace", help="Inspect a local agent run")
    trace_actions = trace_command.add_subparsers(dest="trace_action", required=True)
    trace_show = trace_actions.add_parser("show", help="Show a redacted JSONL run trace")
    trace_show.add_argument("run_id", help="Run ID from an answer's trace_id field")
    trace_show.add_argument("--html", type=Path, help="Write a self-contained HTML viewer")
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
    elif args.command == "chat":
        asyncio.run(_chat(args.repo, settings, args.max_steps, args.token_budget))
    elif args.command == "memory":
        memory_store = MemoryStore(settings.memory_db_path)
        if args.memory_action == "list":
            records = memory_store.list_active(repo=args.repo)
            print(json.dumps([row.model_dump(mode="json") for row in records], indent=2))
        elif args.memory_action == "add":
            try:
                expires_at = _end_of_day(args.expires) if args.expires else None
                record = memory_store.add_user(
                    args.text,
                    kind=args.kind,
                    repo=args.repo,
                    confidence=args.confidence,
                    expires_at=expires_at,
                )
            except ValueError:
                parser.exit(2, "Memory rejected: check text, repo, confidence, and expiry.\n")
            trace = TraceWriter(Path("traces"))
            trace.record(
                "memory_write",
                scope="long_term",
                status="stored",
                record_id=record.id,
                kind=record.kind,
                source=record.source,
                repo=record.repo,
                confidence=record.confidence,
                expires_at=record.expires_at,
            )
            print(record.model_dump_json(indent=2, ensure_ascii=True))
            print(f"Trace: {trace.run_id}", file=sys.stderr)
        elif args.memory_action == "forget":
            if not memory_store.forget(args.id):
                parser.exit(2, "Memory ID not found.\n")
            trace = TraceWriter(Path("traces"))
            trace.record(
                "memory_write", scope="long_term", status="forgotten", record_id=args.id
            )
            print(f"Forgot memory {args.id}")
            print(f"Trace: {trace.run_id}", file=sys.stderr)
    elif args.command == "trace":
        try:
            events = load_trace(Path("traces"), args.run_id)
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            parser.exit(2, f"Cannot show trace: {exc}\n")
        if args.html is not None:
            output = render_trace_html(events, args.run_id, args.html)
            print(f"Wrote trace viewer to {output}")
        else:
            print(json.dumps(events, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
