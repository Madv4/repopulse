"""Bounded LangGraph plan/tool/observe loop with source-linked answers."""

import asyncio
import json
import re
import unicodedata
from collections.abc import Awaitable, Callable
from typing import Any, Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError

from repopulse.github.client import GitHubClient
from repopulse.llm.base import ChatMessage, LLMProvider, ProviderError
from repopulse.mcp_server import Evidence, ToolEvidence
from repopulse.observability.tracer import TraceWriter

TOOL_NAMES = frozenset(
    {
        "search_issues",
        "get_issue",
        "get_pull_request",
        "get_pr_files",
        "list_commits",
        "list_milestones",
        "list_releases",
        "semantic_search",
    }
)
CITABLE_KINDS = frozenset({"issue", "pull_request", "commit"})


class Claim(BaseModel):
    """One answer assertion with at least one retrieved citation."""

    text: str = Field(min_length=1)
    citations: list[str] = Field(min_length=1)


class Answer(BaseModel):
    """Validated answer; absent evidence is expressed as a limitation."""

    summary: Claim | None = None
    key_changes: list[Claim] = Field(default_factory=list)
    risks: list[Claim] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    degraded: bool = False
    degradation_reasons: list[str] = Field(default_factory=list)
    trace_id: str | None = None


class ToolChoice(BaseModel):
    """One allowlisted MCP call selected by the planner."""

    tool: Literal[
        "search_issues",
        "get_issue",
        "get_pull_request",
        "get_pr_files",
        "list_commits",
        "list_milestones",
        "list_releases",
        "semantic_search",
    ]
    args: dict[str, str | int]


class Plan(BaseModel):
    """A short set of evidence-seeking subquestions."""

    subquestions: list[str] = Field(min_length=1, max_length=3)


class Decision(BaseModel):
    """The graph's explicit continue/enough/replan/stop branch."""

    action: Literal["continue", "enough", "replan", "stop"]


class ToolGateway(Protocol):
    """MCP-style tool caller, replaced by recorded responses in tests."""

    async def call(self, name: str, arguments: dict[str, str | int]) -> ToolEvidence: ...


class AgentState(TypedDict):
    question: str
    repo: str
    plan: list[str]
    selected: ToolChoice | None
    evidence: list[Evidence]
    limitations: list[str]
    degradation_reasons: list[str]
    attempted: list[str]
    steps: int
    replans: int
    tokens_used: int
    decision: str
    answer: Answer | None


class BudgetExhausted(Exception):
    """The next LLM call would cross the configured accounting budget."""


def _json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def _supports_claim(claim: Claim, evidence: Evidence) -> bool:
    """A citation supports only text copied from its retrieved title or passage."""
    wanted = _normalized(claim.text)
    return bool(wanted) and any(
        wanted in _normalized(source) for source in (evidence.title, evidence.excerpt)
    )


class RepoPulseAgent:
    """Plan, call MCP tools and synthesize within hard graph limits."""

    def __init__(
        self,
        provider: LLMProvider,
        gateway: ToolGateway,
        *,
        max_steps: int = 8,
        token_budget: int = 8000,
        max_replans: int = 3,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        progress: Callable[[str], None] | None = None,
        trace: TraceWriter | None = None,
    ) -> None:
        if max_steps < 1 or token_budget < 1 or max_replans < 0:
            raise ValueError("agent limits must be positive (max_replans may be zero)")
        self.provider = provider
        self.gateway = gateway
        self.max_steps = max_steps
        self.token_budget = token_budget
        self.max_replans = max_replans
        self.sleep = sleep
        self.progress = progress
        self.trace = trace
        graph = StateGraph(AgentState)
        graph.add_node("plan", self._plan)
        graph.add_node("select", self._select)
        graph.add_node("execute", self._execute)
        graph.add_node("observe", self._observe)
        graph.add_node("decide", self._decide)
        graph.add_node("synthesize", self._synthesize)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "select")
        graph.add_conditional_edges(
            "select", self._after_select, {"execute": "execute", "synthesize": "synthesize"}
        )
        graph.add_edge("execute", "observe")
        graph.add_edge("observe", "decide")
        graph.add_conditional_edges(
            "decide",
            self._after_decide,
            {"select": "select", "plan": "plan", "synthesize": "synthesize"},
        )
        graph.add_edge("synthesize", END)
        self.graph = graph.compile()

    def _degrade(self, state: AgentState, reason: str) -> None:
        if reason not in state["degradation_reasons"]:
            state["degradation_reasons"].append(reason)
            if self.trace is not None:
                self.trace.degradation(reason)

    @staticmethod
    def _provider_failure(stage: str, exc: ProviderError) -> str:
        status = re.search(r"HTTP \d{3}", str(exc))
        detail = status.group() if status else type(exc).__name__
        return f"{stage} LLM request failed: {detail}."

    def _finish_answer(self, state: AgentState, answer: Answer) -> AgentState:
        answer.degradation_reasons = state["degradation_reasons"].copy()
        answer.degraded = bool(answer.degradation_reasons)
        answer.trace_id = self.trace.run_id if answer.degraded and self.trace else None
        state["answer"] = answer
        return state

    def _complete(
        self,
        state: AgentState,
        instruction: str,
        payload: dict[str, Any],
        *,
        output_limit: int = 512,
    ) -> str:
        messages = [
            ChatMessage(role="system", content=instruction),
            ChatMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
        ]
        # Character-based reservation is an accounting estimate, not a provider token count.
        estimate = max(1, sum(len(message.content) for message in messages) // 4)
        remaining = self.token_budget - state["tokens_used"] - estimate
        if remaining < 1:
            raise BudgetExhausted
        response = self.provider.complete(messages, max_output_tokens=min(remaining, output_limit))
        used = max(
            estimate + max(1, len(response.text) // 4),
            response.prompt_tokens + response.completion_tokens,
        )
        state["tokens_used"] += used
        return response.text

    def _plan(self, state: AgentState) -> AgentState:
        if state["plan"]:
            state["replans"] += 1
        try:
            text = self._complete(
                state,
                'Return JSON {"subquestions":["..."]}. Plan up to 3 ways to find '
                "issue, PR or commit evidence.",
                {
                    "question": state["question"],
                    "repo": state["repo"],
                    "previous": state["plan"],
                    "found_urls": [e.url for e in state["evidence"]],
                },
            )
            plan = Plan.model_validate(_json_object(text))
            state["plan"] = plan.subquestions
        except BudgetExhausted:
            state["plan"] = [state["question"]]
            state["decision"] = "stop"
            state["limitations"].append("The token budget cannot fit another LLM call.")
            self._degrade(state, "Planning LLM skipped because the token budget was exhausted.")
        except ProviderError as exc:
            state["plan"] = [state["question"]]
            self._degrade(state, self._provider_failure("Planning", exc))
        except ValidationError:
            state["plan"] = [state["question"]]
            self._degrade(state, "Planning LLM returned invalid structured output.")
        return state

    def _select(self, state: AgentState) -> AgentState:
        if state["decision"] == "stop" or state["steps"] >= self.max_steps:
            state["selected"] = None
            return state
        subquestion = state["plan"][min(len(state["attempted"]), len(state["plan"]) - 1)]
        try:
            text = self._complete(
                state,
                'Return JSON {"tool":"semantic_search","args":{"query":"..."}}. '
                "Allowed tools: "
                + ", ".join(sorted(TOOL_NAMES))
                + ". Use one tool to answer the subquestion.",
                {
                    "question": subquestion,
                    "repo": state["repo"],
                    "attempted_tools": state["attempted"],
                    "evidence_urls": [e.url for e in state["evidence"]],
                },
            )
            choice = ToolChoice.model_validate(_json_object(text))
        except BudgetExhausted:
            state["decision"] = "stop"
            state["selected"] = None
            state["limitations"].append("The token budget cannot fit another LLM call.")
            self._degrade(
                state, "Tool-selection LLM skipped because the token budget was exhausted."
            )
            return state
        except ProviderError as exc:
            self._degrade(state, self._provider_failure("Tool-selection", exc))
            choice = ToolChoice(tool="semantic_search", args={"query": subquestion})
        except ValidationError:
            self._degrade(state, "Tool-selection LLM returned invalid structured output.")
            choice = ToolChoice(tool="semantic_search", args={"query": subquestion})
        choice.args["repo"] = state["repo"]
        state["selected"] = choice
        if state["tokens_used"] >= self.token_budget:
            state["decision"] = "stop"
            state["selected"] = None
        return state

    @staticmethod
    def _after_select(state: AgentState) -> str:
        return "execute" if state["selected"] is not None else "synthesize"

    async def _execute(self, state: AgentState) -> AgentState:
        choice = state["selected"]
        if choice is None:
            return state
        state["attempted"].append(choice.tool)
        for retry in range(3):
            if state["steps"] >= self.max_steps:
                state["limitations"].append("Tool attempt limit reached before retry completed.")
                break
            state["steps"] += 1
            try:
                if self.progress is not None:
                    self.progress(
                        f"Running {choice.tool} (step {state['steps']}/{self.max_steps}, "
                        f"attempt {retry + 1}/3)"
                    )
                result = await self.gateway.call(choice.tool, choice.args)
                known = {(item.kind, item.url, item.excerpt) for item in state["evidence"]}
                state["evidence"].extend(
                    item
                    for item in result.evidence
                    if (item.kind, item.url, item.excerpt) not in known
                )
                break
            except Exception as exc:
                if retry == 2 or state["steps"] >= self.max_steps:
                    state["limitations"].append(
                        f"{choice.tool} failed after {retry + 1} attempt(s): {type(exc).__name__}."
                    )
                else:
                    await self.sleep(float(2**retry))
        return state

    @staticmethod
    def _observe(state: AgentState) -> AgentState:
        # The state holds returned evidence; this node marks the observation boundary.
        return state

    def _decide(self, state: AgentState) -> AgentState:
        if state["steps"] >= self.max_steps or state["tokens_used"] >= self.token_budget:
            state["decision"] = "stop"
            return state
        citable = [e for e in state["evidence"] if e.kind in CITABLE_KINDS]
        try:
            text = self._complete(
                state,
                'Return JSON {"action":"continue|enough|replan|stop"}. '
                "Use replan if evidence is insufficient; enough only when cited "
                "evidence answers the question.",
                {
                    "question": state["question"],
                    "evidence": [e.model_dump() for e in citable[:6]],
                    "limitations": state["limitations"],
                    "replans": state["replans"],
                },
            )
            decision = Decision.model_validate(_json_object(text)).action
        except BudgetExhausted:
            state["limitations"].append("The token budget cannot fit another LLM call.")
            self._degrade(
                state, "Evidence-decision LLM skipped because the token budget was exhausted."
            )
            decision = "stop"
        except ProviderError as exc:
            self._degrade(state, self._provider_failure("Evidence-decision", exc))
            decision = "enough" if citable else "replan"
        except ValidationError:
            self._degrade(state, "Evidence-decision LLM returned invalid structured output.")
            decision = "enough" if citable else "replan"
        if not citable and decision == "enough":
            decision = "replan"
        if decision == "replan" and state["replans"] >= self.max_replans:
            decision = "stop"
            state["limitations"].append("Evidence remained insufficient after 3 replans.")
        if state["tokens_used"] >= self.token_budget:
            decision = "stop"
        state["decision"] = decision
        return state

    @staticmethod
    def _after_decide(state: AgentState) -> str:
        return {"continue": "select", "replan": "plan"}.get(state["decision"], "synthesize")

    def _synthesize(self, state: AgentState) -> AgentState:
        sources: dict[str, list[Evidence]] = {}
        for item in state["evidence"]:
            if item.kind in CITABLE_KINDS:
                sources.setdefault(item.url, []).append(item)
        answer = Answer(limitations=state["limitations"].copy())
        if state["steps"] >= self.max_steps:
            answer.limitations.append("The maximum tool-step limit was reached.")
        if state["tokens_used"] >= self.token_budget:
            answer.limitations.append("The token budget was exhausted.")
        if not sources:
            answer.limitations.append(
                "No citable issue, pull request or commit evidence was retrieved."
            )
            return self._finish_answer(state, answer)
        try:
            text = self._complete(
                state,
                "Return JSON with summary:{text,citations}, key_changes:[{text,citations}], "
                "risks:[{text,citations}]. Every claim must cite retrieved issue, "
                "PR or commit URLs. "
                "Copy claim text exactly from the cited evidence title or excerpt. "
                "Do not paraphrase or invent facts. Omit unsupported claims.",
                {
                    "question": state["question"],
                    "evidence": [
                        e.model_dump()
                        for e in state["evidence"]
                        if e.kind in CITABLE_KINDS
                    ][:8],
                },
                output_limit=1024,
            )
            proposed = Answer.model_validate(_json_object(text))
        except (BudgetExhausted, ProviderError, ValidationError) as exc:
            first = next(iter(sources.values()))[0]
            proposed = Answer(summary=Claim(text=first.title, citations=[first.url]))
            answer.limitations.append(
                "LLM synthesis was unavailable; summary uses a retrieved title."
            )
            reason = (
                self._provider_failure("Synthesis", exc)
                if isinstance(exc, ProviderError)
                else "Synthesis LLM returned invalid structured output."
                if isinstance(exc, ValidationError)
                else "Synthesis LLM skipped because the token budget was exhausted."
            )
            self._degrade(state, reason)

        def grounded(claim: Claim | None) -> Claim | None:
            if claim is None:
                return None
            valid = list(
                dict.fromkeys(
                    url
                    for url in claim.citations
                    if url in sources and any(_supports_claim(claim, item) for item in sources[url])
                )
            )
            return Claim(text=claim.text, citations=valid) if valid else None

        answer.summary = grounded(proposed.summary)
        answer.key_changes = [valid for item in proposed.key_changes if (valid := grounded(item))]
        answer.risks = [valid for item in proposed.risks if (valid := grounded(item))]
        if answer.summary is None and not answer.key_changes and not answer.risks:
            first = next(iter(sources.values()))[0]
            answer.summary = Claim(text=first.title, citations=[first.url])
            answer.limitations.append(
                "No supported LLM claims were returned; summary uses a retrieved title."
            )
            self._degrade(state, "Synthesis claims were unsupported by their cited evidence.")
        if proposed.summary is not None and answer.summary is None:
            answer.limitations.append("An uncited summary was removed.")
        return self._finish_answer(state, answer)

    async def ask(self, question: str, repo: str) -> Answer:
        """Run one bounded graph and return a validated, citation-bearing answer."""
        GitHubClient.repo_path(repo)
        if not question.strip():
            raise ValueError("question must not be empty")
        initial: AgentState = {
            "question": question,
            "repo": repo,
            "plan": [],
            "selected": None,
            "evidence": [],
            "limitations": [],
            "degradation_reasons": [],
            "attempted": [],
            "steps": 0,
            "replans": 0,
            "tokens_used": 0,
            "decision": "continue",
            "answer": None,
        }
        final = await self.graph.ainvoke(initial, config={"recursion_limit": 100})
        result = final["answer"]
        if result is None:
            raise RuntimeError("agent ended without an answer")
        return Answer.model_validate(result)
