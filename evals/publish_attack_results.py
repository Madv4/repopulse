"""Copy measured attack results into docs without hand-editing numbers."""

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
ATTACKS = ROOT / "evals/reports/attacks_live.json"
CLASSIFIER = ROOT / "evals/reports/injection_classifier.json"
EVALUATION = ROOT / "docs/EVALUATION.md"
THREAT_MODEL = ROOT / "docs/THREAT_MODEL.md"


def _replace_block(path: Path, name: str, replacement: str) -> None:
    start = f"<!-- {name}:start -->"
    end = f"<!-- {name}:end -->"
    source = path.read_text(encoding="utf-8")
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError(f"Expected one {name} block in {path}")
    before, remainder = source.split(start, 1)
    _, after = remainder.split(end, 1)
    path.write_text(before + start + "\n" + replacement + "\n" + end + after,
                    encoding="utf-8")


def publish(
    attacks_path: Path = ATTACKS, classifier_path: Path = CLASSIFIER,
    evaluation_path: Path = EVALUATION, threat_path: Path = THREAT_MODEL,
) -> None:
    """Require complete generated reports before publishing their exact counts."""
    attacks: dict[str, Any] = json.loads(attacks_path.read_text(encoding="utf-8"))
    classifier: dict[str, Any] = json.loads(classifier_path.read_text(encoding="utf-8"))
    if attacks.get("metric_source") != "live Groq":
        raise ValueError("Attack report must be from live Groq")
    if "live Groq" not in classifier.get("metric_source", ""):
        raise ValueError("Classifier report must use live Groq")
    attack_rows = attacks["summaries"]
    classifier_rows = classifier["summary"]
    if len(attack_rows) != 4 or len(classifier_rows) != 3:
        raise ValueError("Attack reports do not include every frozen set")
    if any(row["completed"] != row["case_count"] for row in [*attack_rows, *classifier_rows]):
        raise ValueError("Cannot publish incomplete attack results")
    lines = [
        "### Measured live attack behavior (live Groq)", "",
        f"Configured model: `{attacks['model']}`. The optional LLM classifier was off "
        "for both guardrail conditions.", "",
        "| Set | Guardrails | Observed attack successes / cases (live Groq) | "
        "Attack success rate (live Groq) |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in attack_rows:
        lines.append(
            f"| {row['dataset']} | {row['guardrails']} | "
            f"{row['attack_successes']}/{row['case_count']} | "
            f"{row['attack_success_rate']:.3f} |"
        )
    lines += [
        "", "The live scorer observes final-claim leakage, unauthorized executed tools, "
        "and attacker text copied into claims. It can miss subtler compliance. "
        "Per-case behavior labels are in `evals/reports/attacks_live.md`.", "",
        "### Measured optional screening (live Groq classifier; mock LLM agent)", "",
        f"Configured classifier model: `{classifier['model']}`.", "",
        "| Set | Attack comments blocked by heuristic (mock LLM agent) | "
        "Extra attacks blocked by classifier (live Groq) | "
        "Combined blocked / cases (live Groq classifier) | "
        "Combined block rate (live Groq classifier) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in classifier_rows:
        lines.append(
            f"| {row['dataset']} | {row['heuristic_blocked']} | "
            f"{row['extra_classifier_blocks']} | "
            f"{row['combined_blocked']}/{row['case_count']} | "
            f"{row['combined_block_rate']:.3f} |"
        )
    lines += ["", "Attack comments still reaching the mock LLM agent after both screens:"]
    for row in classifier_rows:
        missed = ", ".join(f"`{case_id}`" for case_id in row["missed_case_ids"]) or "none"
        lines.append(f"- {row['dataset']}: {missed}")
    lines += [
        "", "The v2 set was evaluated once after the classifier was committed. "
        "Screening rates are distinct from live attack success rates.",
    ]
    _replace_block(evaluation_path, "measured-attack-results", "\n".join(lines))

    limits = [
        "The generated live Groq behavior report measured the following attack "
        "successes with default guardrails ON versus OFF:"
    ]
    for dataset in ("dev", "heldout_v1"):
        on = next(row for row in attack_rows if row["dataset"] == dataset
                  and row["guardrails"] == "on")
        off = next(row for row in attack_rows if row["dataset"] == dataset
                   and row["guardrails"] == "off")
        limits.append(
            f"- {dataset}: ON {on['attack_successes']}/{on['case_count']} "
            f"(live Groq); OFF {off['attack_successes']}/{off['case_count']} (live Groq)."
        )
    limits.append("The optional classifier plus heuristic left these attack comments unscreened:")
    for row in classifier_rows:
        missed = ", ".join(f"`{case_id}`" for case_id in row["missed_case_ids"]) or "none"
        limits.append(f"- {row['dataset']}: {missed} (live Groq classifier).")
    limits.append(
        "These are observed results on frozen synthetic fixtures, not proof of "
        "general safety. The live behavior scorer can miss paraphrased compliance."
    )
    _replace_block(threat_path, "measured-attack-limits", "\n".join(limits))


if __name__ == "__main__":
    publish()
