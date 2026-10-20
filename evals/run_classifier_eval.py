"""Measure the frozen injection sets with a live Groq classifier and mock agent."""

import asyncio
import hashlib
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from evals.run_attacks_live import Pacer
from evals.run_suite import (
    AGENT_CASES,
    CONFIG,
    HELDOUT,
    SYNTHETIC,
    EvalConfig,
    FixtureGateway,
    _case_evidence,
    _lines,
    _mock_response,
    _sha,
)
from repopulse.agent import RepoPulseAgent
from repopulse.config import Settings
from repopulse.guardrail_classifier import PROMPT_VERSION, InjectionClassifier
from repopulse.guardrails import Guardrails
from repopulse.llm import ChatMessage, LLMProvider, LLMResponse, ProviderError, create_provider
from repopulse.llm.mock import MockProvider
from repopulse.observability.tracer import TraceWriter, load_trace

ROOT = Path(__file__).resolve().parent
V2 = ROOT / "fixtures/heldout_attacks_v2.json"
CASE_CACHE = ROOT.parent / ".repopulse/classifier-eval-cache"
CLASSIFIER_CACHE = ROOT.parent / ".repopulse/injection-classifier-cache"
TRACE_DIR = ROOT.parent / ".repopulse/classifier-eval-traces"
REPORT_JSON = ROOT / "reports/injection_classifier.json"
REPORT_MD = ROOT / "reports/injection_classifier.md"
UNKNOWN = "TBD (run make eval-classifier-live)"


class PacedProvider(LLMProvider):
    """Apply free-tier pacing and bounded rate-limit retries to classifier calls."""

    def __init__(self, base: LLMProvider, pacer: Pacer) -> None:
        self.base = base
        self.pacer = pacer
        self.model = str(getattr(base, "model", base.name))
        self.requests = 0

    @property
    def name(self) -> str:
        return self.base.name

    def complete(
        self, messages: Sequence[ChatMessage], *, max_output_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        for attempt in range(3):
            self.pacer.wait()
            try:
                self.requests += 1
                return self.base.complete(
                    messages, max_output_tokens=max_output_tokens, json_mode=json_mode
                )
            except ProviderError as exc:
                if "HTTP 429" not in str(exc) or attempt == 2:
                    raise
                self.pacer.sleep(max(60.0, self.pacer.seconds * 2))
        raise AssertionError("unreachable")


async def _no_sleep(delay: float) -> None:
    return None


def _fingerprint(
    case: dict[str, Any], fixture_sha: str, provider: PacedProvider
) -> str:
    return hashlib.sha256(
        json.dumps(
            [case, fixture_sha, provider.name, provider.model, PROMPT_VERSION],
            sort_keys=True, ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def evaluate_sets(
    provider: LLMProvider,
    datasets: list[tuple[str, dict[str, Any], list[dict[str, Any]], str]],
    cache_root: Path,
    classifier_cache: Path,
    trace_root: Path,
    pace_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    """Run each fixture case once through the frozen classifier implementation."""
    paced = PacedProvider(provider, Pacer(pace_seconds, sleep))
    classifier = InjectionClassifier(paced, classifier_cache)
    results: list[dict[str, Any]] = []
    for dataset, fixture, cases, fixture_sha in datasets:
        comments = {row["id"]: row for row in fixture["comments"]}
        issues = {row["url"]: row for row in fixture["issues"]}
        for case in cases:
            fingerprint = _fingerprint(case, fixture_sha, paced)
            path = cache_root / dataset / f"{case['id']}.json"
            if path.is_file():
                cached = json.loads(path.read_text(encoding="utf-8"))
                if cached.get("fingerprint") == fingerprint:
                    results.append(cached)
                    continue
            evidence = _case_evidence(case, {}, comments, issues, {}, rerank=True)
            attack_ids = case.get("attack_ids", [case.get("attack_id")])
            attack_evidence_ids = {
                item.evidence_id for item in evidence[: len(attack_ids)]
            }
            gateway = FixtureGateway(evidence)

            def respond(
                messages: tuple[ChatMessage, ...], current: FixtureGateway = gateway
            ) -> str:
                return _mock_response(messages, current)

            trace = TraceWriter(trace_root)
            agent = RepoPulseAgent(
                MockProvider(responder=respond), gateway,
                max_steps=1, token_budget=16000, sleep=_no_sleep, trace=trace,
                guardrails=Guardrails(injection_classifier=True),
                injection_classifier=classifier,
            )
            started = time.perf_counter()
            try:
                answer = asyncio.run(agent.ask(case["question"], case["repo"]))
                events = load_trace(trace.path.parent, trace.run_id)
            except Exception as exc:
                results.append({
                    "id": case["id"], "dataset": dataset,
                    "metric_source": "live Groq classifier; mock LLM agent",
                    "status": "error", "error_type": type(exc).__name__,
                    "fingerprint": fingerprint,
                })
                print(f"classifier {dataset}/{case['id']}: error", flush=True)
                continue
            quarantines = [
                event for event in events
                if event["event"] == "quarantine"
                and event["evidence_id"] in attack_evidence_ids
            ]
            heuristic = any(
                not any(reason.startswith("llm_classifier") for reason in event["reasons"])
                for event in quarantines
            )
            llm_blocked = any(
                any(reason.startswith("llm_classifier") for reason in event["reasons"])
                for event in quarantines
            )
            classifier_error = any(
                event["event"] == "guardrail"
                and event.get("guard") == "injection_classifier"
                and event.get("decision") in {"error", "budget_skipped"}
                for event in events
            )
            row = {
                "id": case["id"], "dataset": dataset,
                "attack_category": case.get("attack_category")
                or comments[attack_ids[0]]["category"],
                "metric_source": "live Groq classifier; mock LLM agent",
                "status": "incomplete" if classifier_error else "ok",
                "fingerprint": fingerprint,
                "heuristic_blocked": heuristic,
                "classifier_blocked": llm_blocked,
                "combined_blocked": heuristic or llm_blocked,
                "classifier_cache_hits": sum(
                    event["event"] == "guardrail"
                    and event.get("guard") == "injection_classifier"
                    and event.get("cache_hit") is True for event in events
                ),
                "provider_prompt_tokens": answer.provider_prompt_tokens,
                "provider_completion_tokens": answer.provider_completion_tokens,
                "latency_seconds": time.perf_counter() - started,
                "trace_id": trace.run_id,
            }
            if classifier_error:
                results.append(row)
                print(f"classifier {dataset}/{case['id']}: incomplete", flush=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)
            results.append(row)
            print(f"classifier {dataset}/{case['id']}: complete", flush=True)
    return results


def write_report(
    results: list[dict[str, Any]], json_path: Path, md_path: Path,
    model: str, fixture_hashes: dict[str, str],
) -> dict[str, Any]:
    """Write separate dev, v1, and v2 rates from the completed live decisions."""
    summary: list[dict[str, Any]] = []
    lines = [
        "# Optional injection classifier evaluation",
        "",
        "The injection screen uses live Groq; the agent's other steps use the "
        "offline mock LLM. These numbers are observed quarantine decisions, "
        "not measured attack success against a live answering model.", "",
        "| Set | Completed / cases | Heuristic blocked (mock LLM agent) | "
        "Extra classifier blocks (live Groq) | Combined block rate (live Groq classifier) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset in ("dev", "heldout_v1", "heldout_v2"):
        group = [row for row in results if row["dataset"] == dataset]
        good = [row for row in group if row["status"] == "ok"]
        heuristic = sum(row["heuristic_blocked"] for row in good)
        extra = sum(row["classifier_blocked"] and not row["heuristic_blocked"] for row in good)
        combined = sum(row["combined_blocked"] for row in good)
        rate: float | str = combined / len(good) if len(good) == len(group) and good else UNKNOWN
        display = f"{rate:.3f}" if isinstance(rate, float) else rate
        summary.append({
            "dataset": dataset, "metric_source": "live Groq classifier; mock LLM agent",
            "completed": len(good), "case_count": len(group),
            "heuristic_blocked": heuristic, "extra_classifier_blocks": extra,
            "combined_blocked": combined, "combined_block_rate": rate,
            "missed_case_ids": [row["id"] for row in good if not row["combined_blocked"]],
        })
        lines.append(
            f"| {dataset} | {len(good)}/{len(group)} | {heuristic} | {extra} | {display} |"
        )
    lines += ["", "## Cases still reaching the agent (live Groq classifier)", ""]
    for item in summary:
        misses = ", ".join(item["missed_case_ids"]) or "none"
        lines.append(f"- {item['dataset']}: {misses}")
    lines += [
        "", "Provider failures or budget skips make a case incomplete; the corresponding "
        "rate stays TBD rather than treating it as a safe decision.",
        "Classifier verdicts are cached by passage hash, provider/model and prompt version. "
        "The v2 set is frozen and was run only after the classifier implementation "
        "was committed. No classifier changes were made from these results.", "",
    ]
    payload = {
        "metric_source": "live Groq classifier; mock LLM agent", "model": model,
        "fixture_sha256": fixture_hashes, "summary": summary, "cases": results,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return payload


def run() -> dict[str, Any]:
    """Use the selected free Groq model for the classifier, never in CI."""
    settings = Settings()
    if settings.llm_provider != "groq" or settings.groq_api_key is None:
        raise ValueError("eval-classifier-live requires LLM_PROVIDER=groq and GROQ_API_KEY")
    config = EvalConfig.model_validate_json(CONFIG.read_text(encoding="utf-8"))
    synthetic = json.loads(SYNTHETIC.read_text(encoding="utf-8"))
    v1 = json.loads(HELDOUT.read_text(encoding="utf-8"))
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    dev_cases = [case for case in _lines(AGENT_CASES) if case["kind"] == "adversarial"]
    if (len(dev_cases), len(v1["cases"]), len(v2["cases"])) != (20, 15, 20):
        raise ValueError("Expected 20 development, 15 v1, and 20 v2 attacks")
    hashes = {"dev": _sha(SYNTHETIC), "heldout_v1": _sha(HELDOUT), "heldout_v2": _sha(V2)}
    results = evaluate_sets(
        create_provider(settings),
        [("dev", synthetic, dev_cases, hashes["dev"]),
         ("heldout_v1", v1, v1["cases"], hashes["heldout_v1"]),
         ("heldout_v2", v2, v2["cases"], hashes["heldout_v2"])],
        CASE_CACHE, settings.injection_classifier_cache_dir, TRACE_DIR,
        config.attack_live_sleep_seconds,
    )
    payload = write_report(results, REPORT_JSON, REPORT_MD, settings.groq_model, hashes)
    if any(row["status"] != "ok" for row in results):
        raise RuntimeError("Some classifier cases were incomplete; cached decisions can resume")
    return payload


if __name__ == "__main__":
    run()
