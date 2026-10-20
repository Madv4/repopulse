"""Build local SVG badges from observed report data and explicit CI state."""

import argparse
import json
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
LATEST = ROOT / "evals/reports/latest.json"
OUT = ROOT / "docs/badges"


def badge(label: str, value: str, color: str) -> str:
    """Render a dependency-free, accessible two-part SVG badge."""
    left = max(55, 15 + len(label) * 7)
    right = max(48, 16 + len(value) * 7)
    width = left + right
    label_xml = escape(label)
    value_xml = escape(value)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="24" '
        f'role="img" aria-label="{escape(label + ": " + value)}">\n'
        f'<title>{label_xml}: {value_xml}</title>\n'
        f'<rect width="{width}" height="24" rx="4" fill="#24292f"/>\n'
        f'<path d="M{left} 0h{right}v24H{left}z" fill="{color}"/>\n'
        f'<text x="{left / 2:.1f}" y="16" fill="#fff" text-anchor="middle" '
        f'font-family="Verdana,Arial,sans-serif" font-size="12">{label_xml}</text>\n'
        f'<text x="{left + right / 2:.1f}" y="16" fill="#fff" text-anchor="middle" '
        f'font-family="Verdana,Arial,sans-serif" font-size="12">{value_xml}</text>\n'
        '</svg>\n'
    )


def eval_badge(report: dict[str, Any]) -> str:
    """Use only the saved mock LLM full-agent pass rate and configured gate."""
    if report.get("metric_source") != "mock LLM":
        raise ValueError("latest report is not labeled mock LLM")
    rate = float(report["full_agent"]["pass_rate"])
    if not 0 <= rate <= 1:
        raise ValueError("invalid full-agent pass rate")
    threshold = float(report["config"]["min_full_agent_pass_rate"])
    passed = rate >= threshold
    if passed != bool(report["ci_pass"]):
        raise ValueError("saved CI gate disagrees with pass rate and threshold")
    return badge("eval · mock LLM", f"{rate * 100:.1f}% pass", "#1a7f37" if passed else "#cf222e")


def ci_badge(status: str) -> str:
    """Render only an explicitly supplied status; default means no remote run."""
    palette = {"pending": "#6e7781", "passing": "#1a7f37", "failing": "#cf222e"}
    if status not in palette:
        raise ValueError("CI status must be pending, passing, or failing")
    value = "awaiting first push" if status == "pending" else status
    return badge("CI", value, palette[status])


def generate(report_path: Path, out_dir: Path, status: str, check_eval: bool = False) -> None:
    """Write badges or fail when the committed eval badge is stale."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = eval_badge(report)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_path = out_dir / "eval.svg"
    if check_eval:
        if not eval_path.is_file() or eval_path.read_text(encoding="utf-8") != expected:
            raise ValueError("eval badge is stale; run `make badges` and commit it")
    else:
        eval_path.write_text(expected, encoding="utf-8")
    (out_dir / "ci.svg").write_text(ci_badge(status), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ci-status", choices=("pending", "passing", "failing"),
                        default="pending")
    parser.add_argument("--check-eval", action="store_true")
    args = parser.parse_args()
    generate(LATEST, OUT, args.ci_status, args.check_eval)
    print(f"Badges saved in {OUT}")
