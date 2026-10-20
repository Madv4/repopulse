"""Render a screenshot-ready HTML trace using a recorded Click issue and mock LLM."""

import asyncio
import json
from pathlib import Path

from repopulse.agent import RepoPulseAgent
from repopulse.llm.base import ChatMessage
from repopulse.llm.mock import MockProvider
from repopulse.mcp_server import Evidence, ToolEvidence
from repopulse.memory import MemoryStore
from repopulse.observability.tracer import TraceWriter, load_trace
from repopulse.observability.viewer import render_trace_html

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "evals/fixtures/click_snapshot.json"
OUTPUT = ROOT / "docs/trace_viewer_example.html"


class RecordedGateway:
    """Return one passage copied from the pinned public Click snapshot."""

    def __init__(self, evidence: Evidence) -> None:
        self.evidence = evidence

    async def call(self, name: str, arguments: dict[str, str | int]) -> ToolEvidence:
        return ToolEvidence(evidence=[self.evidence])


def run() -> Path:
    """Generate an example page without GitHub, a hosted model, or downloads."""
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    row = next(item for item in snapshot["issues"] if item["number"] == 3887)
    evidence = Evidence(
        url=row["html_url"], kind="issue", title=row["title"], excerpt=(row["body"] or "")[:1800]
    )

    def respond(messages: tuple[ChatMessage, ...]) -> str:
        instruction = messages[0].content
        if "subquestions" in instruction:
            return '{"subquestions":["find the ANSI style issue"]}'
        if '"tool"' in instruction:
            return '{"tool":"semantic_search","args":{"query":"ANSI style"}}'
        if '"action"' in instruction:
            return '{"action":"enough"}'
        return json.dumps(
            {
                "summary": {
                    "text": "Click issue 3887 reports ANSI escape codes interfering with style().",
                    "evidence": [
                        {"evidence_id": evidence.evidence_id, "quote": evidence.title}
                    ],
                }
            }
        )

    trace = TraceWriter(ROOT / "traces")
    memory = MemoryStore(ROOT / ".repopulse/trace-example-memory.sqlite3")
    for record in memory.list_active():
        memory.forget(record.id)
    memory.add_user("I prefer concise Click issue summaries.", repo="pallets/click")
    agent = RepoPulseAgent(
        MockProvider(responder=respond), RecordedGateway(evidence),
        trace=trace, long_term_memory=memory,
    )
    asyncio.run(agent.ask("Which issue reports ANSI escape codes breaking style?", "pallets/click"))
    return render_trace_html(load_trace(trace.path.parent, trace.run_id), trace.run_id, OUTPUT)


if __name__ == "__main__":
    print(run())
