"""Offline Phase 6 retrieval and mock-agent evaluation on pinned fixtures."""

import asyncio
import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from evals.metrics import RetrievalCase, aggregate, measure_case
from repopulse.agent import RepoPulseAgent
from repopulse.agent.citations import quote_in_evidence
from repopulse.agent.graph import Claim
from repopulse.config import Settings
from repopulse.guardrails import Guardrails
from repopulse.ingestion.chunking import FixedChunker, StructureChunker
from repopulse.ingestion.embed import LocalEmbedder
from repopulse.ingestion.models import SourceDocument
from repopulse.ingestion.store import ChromaStore
from repopulse.llm.base import ChatMessage
from repopulse.llm.mock import MockProvider
from repopulse.mcp_server import Evidence, ToolEvidence
from repopulse.memory import MemoryStore
from repopulse.observability.tracer import TraceWriter
from repopulse.retrieval.rerank import LocalCrossEncoderReranker
from repopulse.retrieval.retriever import Retriever

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "fixtures/click_large_snapshot.json"
RETRIEVAL_CASES = ROOT / "datasets/retrieval_large.jsonl"
AGENT_CASES = ROOT / "datasets/agent_cases.jsonl"
SYNTHETIC = ROOT / "fixtures/synthetic_repo.json"
CONFIG = ROOT / "config.json"
LATEST_JSON = ROOT / "reports/latest.json"
LATEST_MD = ROOT / "reports/latest.md"


class EvalConfig(BaseModel):
    """CI threshold and bounded evaluation settings."""

    min_full_agent_pass_rate: float = Field(ge=0, le=1)
    retrieval_k: int = Field(ge=1)
    live_case_count: int = Field(ge=1)
    live_sleep_seconds: float = Field(ge=0)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lines(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _documents(snapshot: dict[str, Any]) -> list[SourceDocument]:
    result: list[SourceDocument] = []
    for row in snapshot["issues"]:
        kind: Literal["pull_request", "issue"] = (
            "pull_request" if row.get("pull_request") else "issue"
        )
        result.append(
            SourceDocument(
                repo=snapshot["repo"],
                kind=kind,
                source_id=f"{kind}:{row['number']}",
                number=row["number"],
                url=row["html_url"],
                title=row["title"],
                body=row.get("body") or "",
                author=row["user"]["login"],
                created_at=datetime.fromisoformat(row["created_at"].replace("Z", "+00:00")),
                updated_at=datetime.fromisoformat(row["updated_at"].replace("Z", "+00:00")),
                labels=row["labels"],
            )
        )
    return result


def evaluate_retrieval(
    snapshot: dict[str, Any],
    cases: list[RetrievalCase],
    config: EvalConfig,
    settings: Settings,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[str]]]]:
    """Run four real local embedding/reranker variants with no network calls."""
    documents = _documents(snapshot)
    urls = {document.url for document in documents}
    if len(documents) < 300 or len(cases) < 50:
        raise ValueError("Phase 6 needs at least 300 records and 50 retrieval labels")
    if any(not set(case.relevant_urls) <= urls for case in cases):
        raise ValueError("Retrieval labels reference URLs outside the pinned snapshot")
    embedder = LocalEmbedder(
        settings.model_cache_dir,
        settings.embedding_model,
        revision=settings.embedding_revision,
    )
    reranker = LocalCrossEncoderReranker(
        settings.model_cache_dir,
        settings.reranker_model,
        revision=settings.reranker_revision,
    )
    variants: list[dict[str, Any]] = []
    rankings: dict[str, dict[str, list[str]]] = {}
    for chunker in (FixedChunker(), StructureChunker()):
        store = ChromaStore(settings.chroma_dir.parent / "eval-large", chunker.name)
        store.clear_repo(snapshot["repo"])
        chunks = sum(
            store.replace_document(document, chunker.chunk(document), embedder)
            for document in documents
        )
        retriever = Retriever(store, embedder, reranker)
        for use_reranker in (False, True):
            key = f"{chunker.name}_{'rerank' if use_reranker else 'plain'}"
            start = time.perf_counter()
            by_case: dict[str, list[str]] = {}
            outcomes = []
            for case in cases:
                ranked = [
                    hit.chunk.url
                    for hit in retriever.search(
                        case.query,
                        repo=snapshot["repo"],
                        k=config.retrieval_k,
                        rerank=use_reranker,
                    )
                ]
                by_case[case.id] = ranked
                outcomes.append(measure_case(case, ranked, config.retrieval_k))
            hit_rate, mrr = aggregate(outcomes)
            rankings[key] = by_case
            variants.append(
                {
                    "name": key,
                    "strategy": chunker.name,
                    "reranker": use_reranker,
                    "documents": len(documents),
                    "chunks": chunks,
                    "hit_rate_at_k": hit_rate,
                    "mrr_at_k": mrr,
                    "latency_seconds": time.perf_counter() - start,
                    "cases": [result.model_dump() for result in outcomes],
                }
            )
    return variants, rankings


class FixtureGateway:
    """Return recorded ranked evidence or deterministic simulated HTTP failures."""

    def __init__(self, evidence: list[Evidence], failures: int = 0, status: str = "500") -> None:
        self.evidence = evidence
        self.failures = failures
        self.status = status
        self.calls = 0

    async def call(self, name: str, arguments: dict[str, str | int]) -> ToolEvidence:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError(f"simulated HTTP {self.status}")
        return ToolEvidence(evidence=self.evidence)


def _mock_response(messages: tuple[Any, ...], gateway: FixtureGateway) -> str:
    """Script the model from the prompt text, without access to answer labels."""
    instruction = messages[0].content
    user_text = messages[1].content
    if "subquestions" in instruction:
        question = json.loads(user_text)["question"]
        return json.dumps({"subquestions": [question]})
    if '"tool"' in instruction:
        question = json.loads(user_text)["question"]
        return json.dumps({"tool": "semantic_search", "args": {"query": question}})
    if '"action"' in instruction:
        return json.dumps(
            {
                "action": "enough"
                if gateway.evidence and gateway.calls > gateway.failures
                else "stop"
            }
        )
    match = re.search(
        r'<untrusted_data evidence_id="([0-9a-f]+)">\nEvidence ID: [^\n]+\n'
        r"URL: [^\n]+\nKind: [^\n]+\nTitle:\n([^\n]+)",
        user_text,
    )
    if match:
        evidence_id, title = match.groups()
    else:
        try:
            first = json.loads(user_text)["evidence"][0]
            evidence_id, title = first["evidence_id"], first["title"]
        except (ValueError, IndexError, KeyError, TypeError):
            return '{"summary":null,"key_changes":[],"risks":[]}'
    title = title.replace("\\u003c", "<")
    quote = " ".join(title.split()[:30])
    return json.dumps(
        {
            "summary": {"text": title, "evidence": [{"evidence_id": evidence_id, "quote": quote}]},
            "key_changes": [],
            "risks": [],
        }
    )


def _evidence_from_row(row: dict[str, Any]) -> Evidence:
    return Evidence(
        url=row["html_url"],
        kind="pull_request" if row.get("pull_request") else "issue",
        title=row["title"][:300],
        excerpt=(row.get("body") or "")[:1800],
    )


def _case_evidence(
    case: dict[str, Any],
    rows: dict[str, dict[str, Any]],
    comments: dict[str, dict[str, Any]],
    issues: dict[str, dict[str, Any]],
    rankings: dict[str, dict[str, list[str]]],
    *,
    rerank: bool,
) -> list[Evidence]:
    kind = case["kind"]
    if kind == "adversarial":
        comment = comments[case["attack_id"]]
        issue = issues[comment["parent_url"]]
        return [
            Evidence(
                url=comment["parent_url"],
                kind=comment["kind"],
                title=f"Comment by attacker: {comment['body']}",
                excerpt=comment["body"],
            ),
            Evidence(
                url=issue["url"], kind=issue["kind"], title=issue["title"], excerpt=issue["body"]
            ),
        ]
    if kind == "huge_issue":
        source = rows[case["expected_url"]]
        return [
            Evidence(
                url=source["html_url"],
                kind="issue",
                title=source["title"],
                excerpt=(source["body"] * 100)[:1800],
            )
        ]
    if kind == "tool_failure" and case.get("status") == 500:
        return [_evidence_from_row(rows[case["expected_url"]])]
    if kind != "normal":
        return []
    key = "structure_rerank" if rerank else "structure_plain"
    retrieval_id = f"click-large-{int(case['id'].split('-')[1]):02}"
    return [_evidence_from_row(rows[url]) for url in rankings[key][retrieval_id]]


def _claim_correct(answer: Any, supplied: list[Evidence]) -> bool | None:
    if answer.summary is None:
        return None
    indexed = {item.evidence_id: item for item in supplied}
    return all(
        ref.evidence_id in indexed
        and quote_in_evidence(ref.quote, indexed[ref.evidence_id])
        and indexed[ref.evidence_id].url in answer.summary.citations
        for ref in answer.summary.evidence
    )


def _passed(case: dict[str, Any], summary: Any) -> bool:
    if case.get("expect_refusal"):
        return summary is None
    if summary is None or case.get("expected_url") not in summary.citations:
        return False
    return not (case.get("expected_title") and summary.text != case["expected_title"])


async def _no_sleep(delay: float) -> None:
    return None


def evaluate_agent(
    cases: list[dict[str, Any]],
    snapshot: dict[str, Any],
    synthetic: dict[str, Any],
    rankings: dict[str, dict[str, list[str]]],
    *,
    guardrails: Guardrails | None = None,
    rerank: bool = True,
    memory: bool = False,
) -> dict[str, Any]:
    """Run the actual graph with scripted LLM and recorded/simulated tools."""
    rows = {row["html_url"]: row for row in snapshot["issues"]}
    comments = {row["id"]: row for row in synthetic["comments"]}
    issues = {row["url"]: row for row in synthetic["issues"]}
    results: list[dict[str, Any]] = []
    memory_store = None
    if memory:
        memory_store = MemoryStore(Settings().chroma_dir.parent / "eval-memory.sqlite3")
        for record in memory_store.list_active():
            memory_store.forget(record.id)
        memory_store.add_user("I prefer concise answers about Click issues.", repo="pallets/click")
    for case in cases:
        evidence = _case_evidence(case, rows, comments, issues, rankings, rerank=rerank)
        status = str(case.get("status", "500"))
        failures = (
            2
            if status == "500" and case["kind"] == "tool_failure"
            else (3 if case["kind"] == "tool_failure" else 0)
        )
        gateway = FixtureGateway(evidence, failures=failures, status=status)

        def responder(messages: tuple[ChatMessage, ...], current: FixtureGateway = gateway) -> str:
            return _mock_response(messages, current)

        provider = MockProvider(responder=responder)
        trace = TraceWriter(Settings().chroma_dir.parent / "eval-traces")
        agent = RepoPulseAgent(
            provider,
            gateway,
            max_steps=3,
            token_budget=12000,
            sleep=_no_sleep,
            trace=trace,
            guardrails=guardrails,
            long_term_memory=memory_store,
        )
        started = time.perf_counter()
        answer = asyncio.run(agent.ask(case["question"], case["repo"]))
        elapsed = time.perf_counter() - started
        events = (
            [json.loads(line) for line in trace.path.read_text(encoding="utf-8").splitlines()]
            if trace.path.exists()
            else []
        )
        results.append(
            {
                "id": case["id"],
                "kind": case["kind"],
                "pass": _passed(case, answer.summary),
                "refused": answer.summary is None,
                "citation_correct": _claim_correct(answer, evidence),
                "injection_blocked": any(event["event"] == "quarantine" for event in events),
                "latency_seconds": elapsed,
                "budget_units_used": answer.budget_units_used,
                "provider_prompt_tokens": answer.provider_prompt_tokens,
                "provider_completion_tokens": answer.provider_completion_tokens,
                "tool_calls": gateway.calls,
                "degraded": answer.degraded,
            }
        )
    total = len(results)
    cited = [row["citation_correct"] for row in results if row["citation_correct"] is not None]
    attacks = [row for row in results if row["kind"] == "adversarial"]
    return {
        "case_count": total,
        "pass_rate": sum(row["pass"] for row in results) / total,
        "answer_accuracy": sum(row["pass"] for row in results if row["kind"] == "normal")
        / max(1, sum(row["kind"] == "normal" for row in results)),
        "citation_correctness_rate": sum(bool(value) for value in cited) / len(cited)
        if cited
        else None,
        "injection_block_rate": sum(row["injection_blocked"] for row in attacks) / len(attacks)
        if attacks
        else None,
        "refusal_rate": sum(row["refused"] for row in results) / total,
        "mean_latency_seconds": sum(row["latency_seconds"] for row in results) / total,
        "budget_units_used": sum(row["budget_units_used"] for row in results),
        "provider_prompt_tokens": sum(row["provider_prompt_tokens"] for row in results),
        "provider_completion_tokens": sum(row["provider_completion_tokens"] for row in results),
        "estimated_cost_usd": 0.0,
        "cases": results,
    }


def evaluate_baseline(
    cases: list[dict[str, Any]],
    snapshot: dict[str, Any],
    synthetic: dict[str, Any],
    rankings: dict[str, dict[str, list[str]]],
) -> dict[str, Any]:
    """Single ranked passage, no planning, retry, memory, or guardrail checks."""
    rows = {row["html_url"]: row for row in snapshot["issues"]}
    comments = {row["id"]: row for row in synthetic["comments"]}
    issues = {row["url"]: row for row in synthetic["issues"]}
    results: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        evidence = _case_evidence(case, rows, comments, issues, rankings, rerank=False)
        if case["kind"] == "tool_failure":
            evidence = []
        first = evidence[0] if evidence else None
        summary: Claim | None = None
        budget_units = 0
        prompt_tokens = 0
        completion_tokens = 0
        if first is not None:
            messages = (
                ChatMessage(role="system", content="Answer once using the retrieved passage."),
                ChatMessage(
                    role="user",
                    content=json.dumps(
                        {
                            "question": case["question"],
                            "evidence": [
                                {
                                    "evidence_id": first.evidence_id,
                                    "title": first.title,
                                    "excerpt": first.excerpt,
                                }
                            ],
                        },
                        ensure_ascii=False,
                    ),
                ),
            )
            def baseline_responder(prompt: tuple[ChatMessage, ...]) -> str:
                return _mock_response(prompt, FixtureGateway([]))

            response = MockProvider(responder=baseline_responder).complete(messages)
            proposed = json.loads(response.text).get("summary")
            if proposed is not None:
                summary = Claim.model_validate({**proposed, "citations": [first.url]})
            estimate = max(1, sum(len(message.content) for message in messages) // 4)
            budget_units = max(
                estimate + max(1, len(response.text) // 4),
                response.prompt_tokens + response.completion_tokens,
            )
            prompt_tokens = response.prompt_tokens
            completion_tokens = response.completion_tokens
        results.append(
            {
                "id": case["id"],
                "kind": case["kind"],
                "pass": _passed(case, summary),
                "refused": summary is None,
                "citation_correct": None if summary is None or first is None else all(
                    ref.evidence_id == first.evidence_id
                    and quote_in_evidence(ref.quote, first)
                    and first.url in summary.citations
                    for ref in summary.evidence
                ),
                "injection_blocked": False,
                "latency_seconds": time.perf_counter() - started,
                "budget_units_used": budget_units,
                "provider_prompt_tokens": prompt_tokens,
                "provider_completion_tokens": completion_tokens,
            }
        )
    total = len(results)
    cited = [row["citation_correct"] for row in results if row["citation_correct"] is not None]
    attacks = [row for row in results if row["kind"] == "adversarial"]
    return {
        "case_count": total,
        "pass_rate": sum(row["pass"] for row in results) / total,
        "answer_accuracy": sum(row["pass"] for row in results if row["kind"] == "normal")
        / max(1, sum(row["kind"] == "normal" for row in results)),
        "citation_correctness_rate": sum(bool(value) for value in cited) / len(cited)
        if cited
        else None,
        "injection_block_rate": sum(row["injection_blocked"] for row in attacks) / len(attacks)
        if attacks
        else None,
        "refusal_rate": sum(row["refused"] for row in results) / total,
        "mean_latency_seconds": sum(row["latency_seconds"] for row in results) / total,
        "budget_units_used": sum(row["budget_units_used"] for row in results),
        "provider_prompt_tokens": sum(row["provider_prompt_tokens"] for row in results),
        "provider_completion_tokens": sum(row["provider_completion_tokens"] for row in results),
        "estimated_cost_usd": 0.0,
        "cases": results,
    }


def run() -> dict[str, Any]:
    """Generate JSON/Markdown reports and enforce the configured pass threshold."""
    config = EvalConfig.model_validate_json(CONFIG.read_text(encoding="utf-8"))
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    synthetic = json.loads(SYNTHETIC.read_text(encoding="utf-8"))
    retrieval_cases = [RetrievalCase.model_validate(row) for row in _lines(RETRIEVAL_CASES)]
    agent_cases = _lines(AGENT_CASES)
    if len(agent_cases) < 60 or sum(c["kind"] == "adversarial" for c in agent_cases) < 20:
        raise ValueError("Phase 6 requires 60 cases including 20 attacks")
    variants, rankings = evaluate_retrieval(snapshot, retrieval_cases, config, Settings())
    full = evaluate_agent(agent_cases, snapshot, synthetic, rankings)
    baseline = evaluate_baseline(agent_cases, snapshot, synthetic, rankings)
    guardrails_off = evaluate_agent(
        [c for c in agent_cases if c["kind"] == "adversarial"],
        snapshot,
        synthetic,
        rankings,
        guardrails=Guardrails(
            input_validation=False,
            injection_detection=False,
            untrusted_wrapping=False,
            citation_verification=False,
            output_redaction=False,
            structured_validation=False,
            action_gate=False,
        ),
    )
    normal_cases = [c for c in agent_cases if c["kind"] == "normal"]
    reranker_off = evaluate_agent(normal_cases, snapshot, synthetic, rankings, rerank=False)
    memory_on = evaluate_agent(normal_cases[:10], snapshot, synthetic, rankings, memory=True)
    memory_off = evaluate_agent(normal_cases[:10], snapshot, synthetic, rankings, memory=False)
    payload = {
        "fixtures": {
            "click_sha256": _sha(SNAPSHOT),
            "synthetic_sha256": _sha(SYNTHETIC),
            "retrieval_labels_sha256": _sha(RETRIEVAL_CASES),
            "agent_cases_sha256": _sha(AGENT_CASES),
            "captured_at": snapshot["captured_at"],
            "repo": snapshot["repo"],
        },
        "config": config.model_dump(),
        "retrieval": variants,
        "full_agent": full,
        "naive_rag": baseline,
        "ablations": {
            "guardrails_off": guardrails_off,
            "reranker_off": reranker_off,
            "memory_on": memory_on,
            "memory_off": memory_off,
        },
        "llm": "deterministic mock (offline)",
        "live_model_cost": "TBD (run make eval)",
        "ci_pass": full["pass_rate"] >= config.min_full_agent_pass_rate,
    }
    LATEST_JSON.parent.mkdir(parents=True, exist_ok=True)
    LATEST_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_markdown(payload, config)
    return payload


def _write_markdown(payload: dict[str, Any], config: EvalConfig) -> None:
    """Render only measured values from the just-completed run."""
    lines = [
        "# RepoPulse Phase 6 evaluation",
        "",
        "Offline recorded-fixture run with a deterministic mock LLM. These are not real-model ",
        "answer-accuracy estimates. No GitHub or LLM network calls occur in `make eval`.",
        f"Click snapshot captured: {payload['fixtures']['captured_at']}.",
        f"Recorded issues/PRs: {payload['retrieval'][0]['documents']}.",
        f"Retrieval cases: {len(payload['retrieval'][0]['cases'])}; "
        f"agent cases: {payload['full_agent']['case_count']}.",
        "",
        f"## Retrieval (k={config.retrieval_k})",
        "",
        "| Chunking | Reranker | Chunks | Hit rate@k | MRR@k |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for variant in payload["retrieval"]:
        lines.append(
            f"| {variant['strategy']} | {'on' if variant['reranker'] else 'off'} | "
            f"{variant['chunks']} | {variant['hit_rate_at_k']:.3f} | "
            f"{variant['mrr_at_k']:.3f} |"
        )
    lines += [
        "",
        "## Naive RAG versus full agent",
        "",
        "| System | Cases | Pass rate | Answer accuracy (normal) | Citation correctness | "
        "Injection block | Refusal | Mean latency (s) | Budget units | "
        "Provider tokens (in/out) | Est. cost (USD) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, key in (("Naive RAG", "naive_rag"), ("Full agent", "full_agent")):
        item = payload[key]
        lines.append(
            f"| {label} | {item['case_count']} | {item['pass_rate']:.3f} | "
            f"{item['answer_accuracy']:.3f} | {item['citation_correctness_rate']:.3f} | "
            f"{item['injection_block_rate']:.3f} | {item['refusal_rate']:.3f} | "
            f"{item['mean_latency_seconds']:.4f} | {item['budget_units_used']} | "
            f"{item['provider_prompt_tokens']}/{item['provider_completion_tokens']} | "
            f"{item['estimated_cost_usd']:.2f} |"
        )
    lines += [
        "",
        "## Ablations",
        "",
        "| Variant | Cases | Pass rate | Injection block |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key in (
        ("Guardrails off", "guardrails_off"),
        ("Reranker off", "reranker_off"),
        ("Memory on", "memory_on"),
        ("Memory off", "memory_off"),
    ):
        item = payload["ablations"][key]
        block = item["injection_block_rate"]
        lines.append(
            f"| {label} | {item['case_count']} | {item['pass_rate']:.3f} | {block:.3f} |"
            if block is not None
            else f"| {label} | {item['case_count']} | {item['pass_rate']:.3f} | n/a |"
        )
    lines += [
        "",
        f"CI threshold (configured before run): {config.min_full_agent_pass_rate:.3f}; "
        f"result: {'pass' if payload['ci_pass'] else 'fail'}.",
        "Budget units are the agent's character-based reservation or provider usage, "
        "whichever is larger. Mock responses report no provider tokens.",
        "Mock execution uses no paid service, so its estimated service cost is USD 0. "
        "Provider-reported tokens are zero because the mock does not report token usage. "
        "Real-model cost: TBD (run make eval).",
        "Per-case outcomes, rankings, hashes, and raw measured values are in `latest.json`.",
    ]
    LATEST_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    report = run()
    print(f"Wrote {LATEST_MD} and {LATEST_JSON}; CI pass={report['ci_pass']}")
    if not report["ci_pass"]:
        raise SystemExit(1)
