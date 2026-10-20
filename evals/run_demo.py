"""Offline mock-LLM demo of a caught recorded comment injection and trace."""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from evals.run_suite import AGENT_CASES, SYNTHETIC, FixtureGateway, _case_evidence, _mock_response
from repopulse.agent import RepoPulseAgent
from repopulse.llm import ChatMessage
from repopulse.llm.mock import MockProvider
from repopulse.observability.tracer import TraceWriter, load_trace
from repopulse.observability.viewer import render_trace_html

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class DemoResult:
    """Observed output of one fixture-backed run."""

    summary: str
    citations: list[str]
    quarantined: int
    trace_path: Path
    viewer_path: Path
    degraded: bool


async def _no_sleep(delay: float) -> None:
    return None


def run_demo(trace_dir: Path = ROOT / "traces") -> DemoResult:
    """Run an actual agent with a recorded synthetic comment and mock LLM."""
    fixture = json.loads(SYNTHETIC.read_text(encoding="utf-8"))
    cases = [json.loads(line) for line in AGENT_CASES.read_text(encoding="utf-8").splitlines()]
    case = next(row for row in cases if row["id"] == "attack-01")
    comments = {row["id"]: row for row in fixture["comments"]}
    issues = {row["url"]: row for row in fixture["issues"]}
    evidence = _case_evidence(case, {}, comments, issues, {}, rerank=True)
    gateway = FixtureGateway(evidence)

    def respond(messages: tuple[ChatMessage, ...]) -> str:
        return _mock_response(messages, gateway)

    trace = TraceWriter(trace_dir)
    agent = RepoPulseAgent(
        MockProvider(responder=respond), gateway,
        max_steps=1, token_budget=16000, sleep=_no_sleep, trace=trace,
    )
    answer = asyncio.run(agent.ask(case["question"], case["repo"]))
    events = load_trace(trace.path.parent, trace.run_id)
    quarantined = sum(event["event"] == "quarantine" for event in events)
    if not answer.summary or quarantined < 1:
        raise RuntimeError("demo did not produce a cited answer and quarantine")
    viewer_path = render_trace_html(events, trace.run_id, trace_dir / f"{trace.run_id}.html")
    return DemoResult(
        summary=answer.summary.text,
        citations=answer.summary.citations,
        quarantined=quarantined,
        trace_path=trace.path,
        viewer_path=viewer_path,
        degraded=answer.degraded,
    )


if __name__ == "__main__":
    result = run_demo()
    print("Provider: mock LLM; GitHub: recorded synthetic fixture")
    print(f"Caught injection: {result.quarantined} comment quarantined")
    print(f"Answer: {result.summary}")
    print(f"Citations: {', '.join(result.citations)}")
    print(f"Degraded: {str(result.degraded).lower()}")
    print(f"Trace: {result.trace_path}")
    print(f"HTML viewer: {result.viewer_path}")
