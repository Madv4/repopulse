"""Refresh the README's measured tables from generated JSON reports only."""

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
REPORTS = ROOT / "evals/reports"
START = "<!-- measured-results:start -->"
END = "<!-- measured-results:end -->"


def _rate(value: float) -> str:
    return f"{value:.3f}"


def render(
    offline: dict[str, Any], attacks: dict[str, Any],
    screening: dict[str, Any], benign: dict[str, Any],
) -> str:
    """Build labeled tables using only values present in report objects."""
    if offline["metric_source"] != "mock LLM":
        raise ValueError("offline report must be labeled mock LLM")
    if attacks["metric_source"] != "live Groq":
        raise ValueError("attack behavior report must be labeled live Groq")
    if benign["metric_source"] != "heuristic only; live Groq classifier":
        raise ValueError("benign report has an unexpected metric source")
    lines = [
        START,
        "### Retrieval (mock LLM suite; ranking uses no LLM)", "",
        "| Chunking | Reranker | Chunks | Hit rate@3 | MRR@3 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in offline["retrieval"]:
        lines.append(
            f"| {row['strategy'].title()} | {'On' if row['reranker'] else 'Off'} | "
            f"{row['chunks']} | {_rate(row['hit_rate_at_k'])} | {_rate(row['mrr_at_k'])} |"
        )
    lines.extend([
        "", "### Agent comparison (mock LLM, recorded fixtures)", "",
        "| System | Passed / cases | Pass rate | Citation correctness | Injection block rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for label, key in (("Naive RAG", "naive_rag"), ("Full agent", "full_agent")):
        row = offline[key]
        passed = sum(bool(case["pass"]) for case in row["cases"])
        lines.append(
            f"| {label} | {passed}/{row['case_count']} | {_rate(row['pass_rate'])} | "
            f"{_rate(row['citation_correctness_rate'])} | "
            f"{_rate(row['injection_block_rate'])} |"
        )
    lines.extend([
        "", "### Attack screening (heuristic only; live Groq classifier with mock LLM agent)", "",
        "| Set | Heuristic-only blocks | Added live Groq classifier blocks | "
        "Combined live Groq classifier block rate |",
        "| --- | ---: | ---: | ---: |",
    ])
    for row in screening["summary"]:
        lines.append(
            f"| {row['dataset']} | {row['heuristic_blocked']}/{row['case_count']} | "
            f"{row['extra_classifier_blocks']} | "
            f"{row['combined_blocked']}/{row['case_count']} "
            f"({_rate(row['combined_block_rate'])}) |"
        )
    counts = benign["counts"]
    model = benign["classifier_model"]
    lines.extend([
        "", f"### Benign-comment false positives (live Groq classifier `{model}`)", "",
        "| Screen | Quarantined / sample | False-positive proxy rate |",
        "| --- | ---: | ---: |",
    ])
    for label, key in (
        ("Heuristic only", "heuristic_only_mode"),
        ("Live Groq classifier only", "classifier_only_mode"),
        ("Heuristic + live Groq classifier", "both_screens_mode"),
    ):
        count = counts[key]
        sample = counts["sample"]
        lines.append(f"| {label} | {count}/{sample} | {_rate(count / sample)} |")
    lines.extend([
        "", "### Observable attack success (live Groq answering model)", "",
        "| Set | Guardrails | Successful / cases | Attack success rate |",
        "| --- | --- | ---: | ---: |",
    ])
    for row in attacks["summaries"]:
        lines.append(
            f"| {row['dataset']} | {row['guardrails']} | "
            f"{row['attack_successes']}/{row['case_count']} | "
            f"{_rate(row['attack_success_rate'])} |"
        )
    off = [row for row in attacks["summaries"] if row["guardrails"] == "off"]
    if off and all(row["attack_successes"] == 0 for row in off):
        lines.extend([
            "", "The unguarded live Groq run also resisted every attack observed by this "
            "scorer. Its checks can miss subtle or paraphrased compliance.",
        ])
    lines.append(END)
    return "\n".join(lines)


def publish(readme: Path = README, reports: Path = REPORTS) -> None:
    """Replace the marked README block, requiring every measured source."""
    files = {
        name: json.loads((reports / path).read_text(encoding="utf-8"))
        for name, path in (
            ("offline", "latest.json"), ("attacks", "attacks_live.json"),
            ("screening", "injection_classifier.json"), ("benign", "benign_fp.json"),
        )
    }
    content = readme.read_text(encoding="utf-8")
    if content.count(START) != 1 or content.count(END) != 1:
        raise ValueError("README needs one measured-results marker pair")
    start = content.index(START)
    end = content.index(END, start) + len(END)
    updated = content[:start] + render(**files) + content[end:]
    readme.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    publish()
    print(f"Updated measured README tables from {REPORTS}")
