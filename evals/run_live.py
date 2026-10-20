"""Optional ten-question Groq free-tier smoke evaluation with pacing."""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from evals.run_suite import CONFIG, RETRIEVAL_CASES, EvalConfig, _lines
from repopulse.config import Settings

ROOT = Path(__file__).resolve().parent
LIVE_JSON = ROOT / "reports/live_latest.json"
LIVE_MD = ROOT / "reports/live_latest.md"


def run() -> dict[str, Any]:
    """Run bounded real-provider cases; do not use this path in CI or offline eval."""
    settings = Settings()
    if settings.llm_provider != "groq" or settings.groq_api_key is None:
        raise ValueError("eval-live requires configured LLM_PROVIDER=groq and GROQ_API_KEY")
    config = EvalConfig.model_validate_json(CONFIG.read_text(encoding="utf-8"))
    cases = _lines(RETRIEVAL_CASES)[: config.live_case_count]
    if len(cases) != 10:
        raise ValueError("eval-live requires exactly ten labeled cases")
    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        if index:
            time.sleep(config.live_sleep_seconds)
        answer: dict[str, Any] = {}
        failure = ""
        elapsed = 0.0
        for attempt in range(2):
            answer = {}
            started = time.perf_counter()
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "repopulse.cli",
                    "ask",
                    case["query"],
                    "--repo",
                    "pallets/click",
                    "--max-steps",
                    "3",
                    "--token-budget",
                    "10000",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            elapsed += time.perf_counter() - started
            try:
                decoded = json.loads(completed.stdout)
                answer = decoded if isinstance(decoded, dict) else {}
            except json.JSONDecodeError:
                failure = "invalid_cli_output"
            if completed.returncode == 0 and answer:
                failure = ""
                break
            if "HTTP 429" in completed.stderr and attempt == 0:
                time.sleep(max(config.live_sleep_seconds, 60))
                continue
            failure = "provider_or_cli_failure" if answer else "invalid_cli_output"
            break
        summary = answer.get("summary")
        urls = summary.get("citations", []) if isinstance(summary, dict) else []
        results.append(
            {
                "id": case["id"],
                "metric_source": "live Groq",
                "question": case["query"],
                "failure": failure,
                "answer_correct_url": bool(set(urls) & set(case["relevant_urls"])),
                "degraded": answer.get("degraded"),
                "refused": summary is None,
                "latency_seconds": elapsed,
                "provider_prompt_tokens": answer.get("provider_prompt_tokens"),
                "provider_completion_tokens": answer.get("provider_completion_tokens"),
                "estimated_cost": "TBD (run make eval)",
            }
        )
        print(f"Live case {index + 1}/{len(cases)} complete", flush=True)
    payload = {
        "metric_source": "live Groq",
        "provider": "groq",
        "model": settings.groq_model,
        "case_count": len(results),
        "answer_correct_url_rate": sum(row["answer_correct_url"] for row in results) / len(results),
        "non_degraded_rate": sum(row["degraded"] is False for row in results) / len(results),
        "refusal_rate": sum(row["refused"] for row in results) / len(results),
        "mean_latency_seconds": sum(row["latency_seconds"] for row in results) / len(results),
        "provider_prompt_tokens": sum(row["provider_prompt_tokens"] or 0 for row in results),
        "provider_completion_tokens": sum(
            row["provider_completion_tokens"] or 0 for row in results
        ),
        "estimated_cost": "TBD (run make eval)",
        "cases": results,
    }
    LIVE_JSON.parent.mkdir(parents=True, exist_ok=True)
    LIVE_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    LIVE_MD.write_text(
        "# Live Groq evaluation (live Groq)\n\n"
        f"Cases (live Groq): {payload['case_count']}\n\n"
        f"Correct labeled URL rate (live Groq): {payload['answer_correct_url_rate']:.3f}\n\n"
        f"Non-degraded rate (live Groq): {payload['non_degraded_rate']:.3f}\n\n"
        f"Refusal rate (live Groq): {payload['refusal_rate']:.3f}\n\n"
        f"Mean latency (live Groq, s): {payload['mean_latency_seconds']:.3f}\n\n"
        f"Provider prompt tokens (live Groq): {payload['provider_prompt_tokens']}\n\n"
        f"Provider completion tokens (live Groq): {payload['provider_completion_tokens']}\n\n"
        "Estimated cost (live Groq): TBD (run make eval)\n\n"
        "This optional live run uses free-tier pacing and is not part of CI. "
        "URL correctness is a rule-based label check, not semantic answer grading.\n",
        encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    run()
