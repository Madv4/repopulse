"""Offline agent behavior tests using scripted LLM output and recorded evidence."""

import asyncio
import json
from pathlib import Path

from repopulse.agent import RepoPulseAgent
from repopulse.llm.base import LLMResponse
from repopulse.llm.mock import MockProvider
from repopulse.mcp_server import Evidence, ToolEvidence
from repopulse.observability.tracer import TraceWriter

FIXTURE = Path(__file__).resolve().parents[1] / "evals/fixtures/click_snapshot.json"
ISSUE = next(
    row
    for row in json.loads(FIXTURE.read_text(encoding="utf-8"))["issues"]
    if "pull_request" not in row
)
EVIDENCE = Evidence(
    url=ISSUE["html_url"], kind="issue", title=ISSUE["title"], excerpt=ISSUE["body"][:600]
)


class FakeGateway:
    """Recorded MCP result or a configured sequence of tool failures."""

    def __init__(self, failures: int = 0, *, empty: bool = False) -> None:
        self.failures = failures
        self.empty = empty
        self.calls: list[tuple[str, dict[str, str | int]]] = []

    async def call(self, name: str, arguments: dict[str, str | int]) -> ToolEvidence:
        self.calls.append((name, arguments))
        if len(self.calls) <= self.failures:
            raise RuntimeError("fixture tool failure")
        return ToolEvidence(evidence=[] if self.empty else [EVIDENCE])


def run(agent: RepoPulseAgent) -> object:
    return asyncio.run(agent.ask("What happened to ANSI colors?", "pallets/click"))


def test_replans_three_times_when_evidence_is_insufficient() -> None:
    script: list[str] = []
    for index in range(4):
        script += [
            json.dumps({"subquestions": [f"attempt {index}"]}),
            json.dumps({"tool": "semantic_search", "args": {"query": f"attempt {index}"}}),
        ]
        if index < 3:
            script.append('{"action":"replan"}')
        else:
            script.append('{"action":"replan"}')
    provider = MockProvider(script)
    gateway = FakeGateway(empty=True)
    answer = run(RepoPulseAgent(provider, gateway, max_steps=12))
    assert len(gateway.calls) == 4
    assert len([call for call in provider.calls if "subquestions" in call[0].content]) == 4
    assert answer.summary is None
    assert any("3 replans" in item for item in answer.limitations)


def test_hard_step_cap_counts_tool_calls() -> None:
    provider = MockProvider(
        [
            '{"subquestions":["find issue"]}',
            '{"tool":"semantic_search","args":{"query":"ANSI"}}',
            '{"action":"continue"}',
            '{"tool":"get_issue","args":{"number":3887}}',
        ]
    )
    gateway = FakeGateway()
    answer = run(RepoPulseAgent(provider, gateway, max_steps=2))
    assert len(gateway.calls) == 2
    assert any("step limit" in item for item in answer.limitations)
    assert answer.summary is None
    assert answer.degraded


def test_token_budget_blocks_further_llm_and_tool_calls() -> None:
    provider = MockProvider(
        [LLMResponse(text='{"subquestions":["find issue"]}', model="mock", prompt_tokens=999)]
    )
    gateway = FakeGateway()
    answer = run(RepoPulseAgent(provider, gateway, token_budget=300))
    assert len(provider.calls) == 1
    assert gateway.calls == []
    assert answer.summary is None
    assert any("token budget" in item.lower() for item in answer.limitations)


def test_tool_retries_with_backoff_then_continues_with_limitation() -> None:
    provider = MockProvider(
        [
            '{"subquestions":["find issue"]}',
            '{"tool":"semantic_search","args":{"query":"ANSI"}}',
            '{"action":"replan"}',
            '{"subquestions":["try direct issue"]}',
            '{"tool":"get_issue","args":{"number":3887}}',
            '{"action":"enough"}',
            json.dumps(
                {
                    "summary": {"text": ISSUE["title"], "citations": [EVIDENCE.url]},
                    "key_changes": [],
                    "risks": [],
                }
            ),
        ]
    )
    gateway = FakeGateway(failures=3)
    waits: list[float] = []

    async def record_sleep(delay: float) -> None:
        waits.append(delay)

    answer = run(RepoPulseAgent(provider, gateway, sleep=record_sleep))
    assert len(gateway.calls) == 4
    assert waits == [1.0, 2.0]
    assert any("semantic_search failed after 3" in item for item in answer.limitations)
    assert answer.summary is not None
    assert answer.summary.citations == [EVIDENCE.url]


def test_retry_cannot_cross_step_cap() -> None:
    provider = MockProvider([
        '{"subquestions":["find issue"]}',
        '{"tool":"semantic_search","args":{"query":"ANSI"}}',
    ])
    gateway = FakeGateway(failures=99)
    waits: list[float] = []

    async def record_sleep(delay: float) -> None:
        waits.append(delay)

    answer = run(RepoPulseAgent(provider, gateway, max_steps=2, sleep=record_sleep))
    assert len(gateway.calls) == 2
    assert waits == [1.0]
    assert any("failed after 2 attempt" in item for item in answer.limitations)


def test_every_final_claim_cites_retrieved_evidence() -> None:
    bad_url = "https://github.com/pallets/click/issues/999999"
    provider = MockProvider(
        [
            '{"subquestions":["find issue"]}',
            '{"tool":"semantic_search","args":{"query":"ANSI"}}',
            '{"action":"enough"}',
            json.dumps(
                {
                    "summary": {"text": ISSUE["title"], "citations": [EVIDENCE.url]},
                    "key_changes": [
                        {"text": EVIDENCE.title, "citations": [EVIDENCE.url, bad_url]}
                    ],
                    "risks": [{"text": "uncited", "citations": [bad_url]}],
                }
            ),
        ]
    )
    answer = run(RepoPulseAgent(provider, FakeGateway()))
    claims = [answer.summary, *answer.key_changes, *answer.risks]
    assert all(claim is not None and claim.citations == [EVIDENCE.url] for claim in claims)
    assert len(answer.risks) == 0


def test_default_mock_still_works_offline() -> None:
    answer = run(RepoPulseAgent(MockProvider(), FakeGateway()))
    assert answer.summary is None
    assert answer.degraded


def test_claim_requires_support_from_its_own_citation() -> None:
    unrelated = next(
        row
        for row in json.loads(FIXTURE.read_text(encoding="utf-8"))["issues"]
        if row["number"] == 3886
    )
    wrong = Evidence(
        url=unrelated["html_url"],
        kind="issue",
        title=unrelated["title"],
        excerpt=unrelated["body"][:600],
    )

    class MixedGateway(FakeGateway):
        async def call(self, name: str, arguments: dict[str, str | int]) -> ToolEvidence:
            return ToolEvidence(evidence=[EVIDENCE, wrong])

    provider = MockProvider(
        [
            '{"subquestions":["find ANSI color issue"]}',
            '{"tool":"semantic_search","args":{"query":"ANSI color"}}',
            '{"action":"enough"}',
            json.dumps({"summary": {"text": EVIDENCE.title, "citations": [wrong.url]}}),
        ]
    )
    answer = asyncio.run(
        RepoPulseAgent(provider, MixedGateway()).ask(
            "Which issue reported ANSI color handling?", "pallets/click"
        )
    )
    assert answer.summary is None
    assert answer.degraded
    assert any("unsupported" in reason for reason in answer.degradation_reasons)


def test_llm_fallback_marks_answer_degraded_and_traces_reason(tmp_path: Path) -> None:
    from repopulse.llm.base import ProviderError

    def fail(_messages: object) -> str:
        raise ProviderError("LLM service returned HTTP 404")

    trace = TraceWriter(tmp_path / "traces")
    answer = run(
        RepoPulseAgent(MockProvider(responder=fail), FakeGateway(), max_steps=1, trace=trace)
    )
    assert answer.degraded is True
    assert answer.trace_id == trace.run_id
    assert any("HTTP 404" in reason for reason in answer.degradation_reasons)
    events = [json.loads(line) for line in trace.path.read_text(encoding="utf-8").splitlines()]
    assert any(
        event["event"] == "degradation" and "HTTP 404" in event["reason"]
        for event in events
    )
