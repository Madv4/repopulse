"""Measure false quarantines on recorded real Click comments.

The fixture is captured separately. This command uses live Groq only for the
optional classifier and saves decisions by content hash for safe resumption.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from evals.run_attacks_live import Pacer
from evals.run_classifier_eval import PacedProvider
from repopulse.config import Settings
from repopulse.guardrail_classifier import PROMPT_VERSION, InjectionClassifier
from repopulse.guardrails import detect_injection
from repopulse.llm import create_provider
from repopulse.mcp_server import Evidence

ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / "fixtures/click_benign_comments.json"
PARENT = ROOT / "fixtures/click_large_snapshot.json"
REPORT_JSON = ROOT / "reports/benign_fp.json"
REPORT_MD = ROOT / "reports/benign_fp.md"


def evidence_for(row: dict[str, Any]) -> Evidence:
    """Represent one comment with its real parent title and bounded body."""
    return Evidence(
        url=str(row["url"]), kind=str(row["kind"]),
        title=str(row["issue_title"]), excerpt=str(row["body"])[:1800],
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count quarantines in three modes and the exclusive overlap."""
    heuristic = sum(bool(row["heuristic_blocked"]) for row in rows)
    classifier = sum(bool(row["classifier_blocked"]) for row in rows)
    both = sum(bool(row["heuristic_blocked"] and row["classifier_blocked"]) for row in rows)
    return {
        "sample": len(rows),
        "heuristic_only_mode": heuristic,
        "classifier_only_mode": classifier,
        "both_screens_mode": heuristic + classifier - both,
        "overlap": both,
        "heuristic_exclusive": heuristic - both,
        "classifier_exclusive": classifier - both,
    }


def evaluate(settings: Settings, pace_seconds: float = 6.0) -> dict[str, Any]:
    """Run all recorded comments through independent heuristic and live screens."""
    source_bytes = FIXTURE.read_bytes()
    source = json.loads(source_bytes)
    if len(source["comments"]) < 200:
        raise ValueError("benign comment fixture needs at least 200 comments")
    if source["parent_snapshot_sha256"] != hashlib.sha256(PARENT.read_bytes()).hexdigest():
        raise ValueError("parent snapshot hash does not match recorded comment fixture")
    if settings.llm_provider != "groq":
        raise ValueError("false-positive classifier audit requires LLM_PROVIDER=groq")
    provider = PacedProvider(create_provider(settings), Pacer(pace_seconds, time.sleep))
    classifier = InjectionClassifier(provider, settings.injection_classifier_cache_dir)
    rows: list[dict[str, Any]] = []
    for index, comment in enumerate(source["comments"], start=1):
        evidence = evidence_for(comment)
        heuristic = detect_injection(evidence)
        classified = classifier.classify(evidence)
        rows.append({
            "comment_id": comment["id"],
            "comment_url": comment["url"],
            "issue_number": comment["issue_number"],
            "issue_title": comment["issue_title"],
            "kind": comment["kind"],
            "heuristic_blocked": heuristic.suspicious,
            "heuristic_reasons": heuristic.reasons,
            "classifier_blocked": classified.suspicious,
            "classifier_reason": classified.reason,
            "classifier_chunk_hash": classified.chunk_hash,
            "classifier_cache_hit": classified.cache_hit,
            "provider_prompt_tokens": classified.provider_prompt_tokens,
            "provider_completion_tokens": classified.provider_completion_tokens,
        })
        if index % 20 == 0:
            print(f"Reviewed {index}/{len(source['comments'])} recorded comments", flush=True)
    totals = summarize(rows)
    report = {
        "metric_source": "heuristic only; live Groq classifier",
        "classifier_model": settings.groq_model,
        "classifier_prompt_version": PROMPT_VERSION,
        "comment_fixture_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "parent_snapshot_sha256": source["parent_snapshot_sha256"],
        "parent_snapshot_captured_at": source["parent_snapshot_captured_at"],
        "comment_capture_at": source["captured_at"],
        "selection": source["selection"],
        "distinct_parents": len({row["issue_number"] for row in rows}),
        "counts": totals,
        "cases": rows,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    flagged = [row for row in rows if row["heuristic_blocked"] or row["classifier_blocked"]]
    lines = [
        "# Benign Click comment false-positive audit", "",
        "The sample contains real public issue/PR comments on the 400 issue/PR IDs in the "
        "pinned Click fixture. There are no planted attack comments. This is a benign-content "
        "proxy: provenance alone cannot prove every public comment harmless. The classifier "
        "was not tuned on this sample. Each comment is reviewed independently using the real "
        "parent title and the first 1,800 body characters.", "",
        f"- Parent snapshot SHA-256: `{report['parent_snapshot_sha256']}`",
        f"- Comment fixture SHA-256: `{report['comment_fixture_sha256']}`",
        f"- Comments: {totals['sample']}; distinct parents: {report['distinct_parents']}",
        f"- Live Groq classifier model: `{settings.groq_model}`; prompt: `{PROMPT_VERSION}`", "",
        "| Screen mode | Quarantined / sample | False-positive proxy rate |",
        "| --- | ---: | ---: |",
        f"| Heuristic only | {totals['heuristic_only_mode']}/{totals['sample']} | "
        f"{totals['heuristic_only_mode'] / totals['sample']:.3f} |",
        f"| Live Groq classifier only | {totals['classifier_only_mode']}/{totals['sample']} | "
        f"{totals['classifier_only_mode'] / totals['sample']:.3f} |",
        f"| Heuristic + live Groq classifier (union) | "
        f"{totals['both_screens_mode']}/{totals['sample']} | "
        f"{totals['both_screens_mode'] / totals['sample']:.3f} |",
        "", f"Overlap: {totals['overlap']}; heuristic-exclusive: "
        f"{totals['heuristic_exclusive']}; classifier-exclusive: "
        f"{totals['classifier_exclusive']}.", "",
        f"Flagged examples available for manual answer-quality review: {len(flagged)}. "
        "Ten examples cannot be listed unless at least ten are actually observed.", "",
        "| Comment | Heuristic only | Live Groq classifier |",
        "| --- | --- | --- |",
    ]
    for row in flagged[:10]:
        lines.append(
            f"| [#{row['comment_id']}]({row['comment_url']}) on "
            f"#{row['issue_number']} | "
            f"{', '.join(row['heuristic_reasons']) or 'clear'} | "
            f"{row['classifier_reason'] if row['classifier_blocked'] else 'clear'} |"
        )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    settings = Settings(groq_model=str(config["benign_fp_live_model"]))
    result = evaluate(settings, float(config["benign_fp_live_sleep_seconds"]))
    print(f"Benign audit counts: {result['counts']}")
