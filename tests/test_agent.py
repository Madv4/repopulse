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
                    "summary": {
                        "text": ISSUE["title"],
                        "evidence": [
                            {"evidence_id": EVIDENCE.evidence_id, "quote": EVIDENCE.title}
                        ],
                    },
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
    provider = MockProvider(
        [
            '{"subquestions":["find issue"]}',
            '{"tool":"semantic_search","args":{"query":"ANSI"}}',
        ]
    )
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
                    "summary": {
                        "text": ISSUE["title"],
                        "evidence": [
                            {"evidence_id": EVIDENCE.evidence_id, "quote": EVIDENCE.title}
                        ],
                    },
                    "key_changes": [
                        {
                            "text": EVIDENCE.title,
                            "evidence": [
                                {"evidence_id": EVIDENCE.evidence_id, "quote": EVIDENCE.title}
                            ],
                        }
                    ],
                    "risks": [
                        {
                            "text": "uncited",
                            "evidence": [{"evidence_id": bad_url, "quote": "fiction"}],
                        }
                    ],
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
            json.dumps(
                {
                    "summary": {
                        "text": EVIDENCE.title,
                        "evidence": [{"evidence_id": wrong.evidence_id, "quote": EVIDENCE.title}],
                    }
                }
            ),
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
        event["event"] == "degradation" and "HTTP 404" in event["reason"] for event in events
    )


def test_paraphrased_claim_with_valid_quote_passes() -> None:
    provider = MockProvider(
        [
            '{"subquestions":["find ANSI issue"]}',
            '{"tool":"semantic_search","args":{"query":"ANSI"}}',
            '{"action":"enough"}',
            json.dumps(
                {
                    "summary": {
                        "text": "The report concerned how Click handled ANSI colors.",
                        "evidence": [
                            {"evidence_id": EVIDENCE.evidence_id, "quote": EVIDENCE.title}
                        ],
                    }
                }
            ),
        ]
    )
    answer = run(RepoPulseAgent(provider, FakeGateway()))
    assert answer.summary is not None
    assert answer.summary.text != EVIDENCE.title
    assert answer.summary.citations == [EVIDENCE.url]


def test_fabricated_quote_fails() -> None:
    answer_text = json.dumps(
        {
            "summary": {
                "text": "An invented claim.",
                "evidence": [
                    {
                        "evidence_id": EVIDENCE.evidence_id,
                        "quote": "This fabricated sentence is not in the issue.",
                    }
                ],
            }
        }
    )
    provider = MockProvider(
        [
            '{"subquestions":["find ANSI issue"]}',
            '{"tool":"semantic_search","args":{"query":"ANSI"}}',
            '{"action":"enough"}',
            answer_text,
            answer_text,
        ]
    )
    answer = run(RepoPulseAgent(provider, FakeGateway()))
    assert answer.summary is None
    assert answer.degraded


def test_quote_normalizes_line_breaks_but_not_fabrication() -> None:
    from repopulse.agent.citations import quote_in_evidence

    source = Evidence(
        url=EVIDENCE.url,
        kind="issue",
        title="café report",
        excerpt="Shell completion can execute parameter callbacks while Click is only trying to\n"
        "determine completion information.",
    )
    assert quote_in_evidence(
        "Shell completion can execute parameter callbacks while Click is only trying to "
        "determine completion information.",
        source,
    )
    assert quote_in_evidence("cafe\u0301 report", source)
    assert not quote_in_evidence("The issue says all callbacks are disabled.", source)


def test_synthesis_retries_once_with_failed_quote_feedback(tmp_path: Path) -> None:
    trace = TraceWriter(tmp_path / "traces")
    bad = json.dumps(
        {
            "summary": {
                "text": "The ANSI issue was reported.",
                "evidence": [{"evidence_id": EVIDENCE.evidence_id, "quote": "fabricated quote"}],
            }
        }
    )
    good = json.dumps(
        {
            "summary": {
                "text": "The ANSI issue was reported.",
                "evidence": [{"evidence_id": EVIDENCE.evidence_id, "quote": EVIDENCE.title}],
            }
        }
    )
    provider = MockProvider(
        [
            '{"subquestions":["find ANSI issue"]}',
            '{"tool":"semantic_search","args":{"query":"ANSI"}}',
            '{"action":"enough"}',
            bad,
            good,
        ]
    )
    answer = run(RepoPulseAgent(provider, FakeGateway(), trace=trace))
    assert answer.summary is not None
    assert not answer.degraded
    assert "quote_not_in_evidence" in provider.calls[-1][-1].content
    events = [json.loads(line) for line in trace.path.read_text(encoding="utf-8").splitlines()]
    rejection = next(event for event in events if event["event"] == "claim_rejection")
    assert rejection["quote"] == "fabricated quote"
    assert EVIDENCE.title in rejection["evidence_text"]


def test_planning_json_mode_http_400_retries_without_degrading() -> None:
    from repopulse.llm.base import ProviderError

    planning_attempts = 0

    def respond(messages: object) -> str:
        nonlocal planning_attempts
        assert isinstance(messages, tuple)
        instruction = messages[0].content
        if "subquestions" in instruction:
            planning_attempts += 1
            if planning_attempts == 1:
                raise ProviderError("LLM service returned HTTP 400")
            return '{"subquestions":["find ANSI issue"]}'
        if '"tool"' in instruction:
            return '{"tool":"semantic_search","args":{"query":"ANSI"}}'
        if '"action"' in instruction:
            return '{"action":"enough"}'
        return json.dumps(
            {
                "summary": {
                    "text": "The ANSI issue was reported.",
                    "evidence": [{"evidence_id": EVIDENCE.evidence_id, "quote": EVIDENCE.title}],
                }
            }
        )

    answer = run(RepoPulseAgent(MockProvider(responder=respond), FakeGateway()))
    assert planning_attempts == 2
    assert answer.summary is not None
    assert not answer.degraded


def test_planner_retries_validation_with_error_and_traces_raw(tmp_path: Path) -> None:
    trace = TraceWriter(tmp_path / "traces")
    provider = MockProvider(
        [
            '```json\n{"subquestions":[]}\n```',
            '```json\n{"subquestions":["find ANSI issue"]}\n```',
            '{"tool":"semantic_search","args":{"query":"ANSI"}}',
            '{"action":"enough"}',
            json.dumps(
                {
                    "summary": {
                        "text": "ANSI issue reported",
                        "evidence": [
                            {"evidence_id": EVIDENCE.evidence_id, "quote": EVIDENCE.title}
                        ],
                    }
                }
            ),
        ]
    )
    answer = run(RepoPulseAgent(provider, FakeGateway(), trace=trace))
    assert answer.summary is not None
    assert not any("Planning" in reason for reason in answer.degradation_reasons)
    assert "Fix your JSON" in provider.calls[1][-1].content
    events = [json.loads(line) for line in trace.path.read_text(encoding="utf-8").splitlines()]
    failure = next(event for event in events if event["event"] == "validation_failure")
    assert failure["stage"] == "planning"
    assert failure["raw_output"].startswith("```json")
