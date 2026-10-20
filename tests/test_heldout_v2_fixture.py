"""Structural checks only; v2 behavior evaluation is a one-time live run."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "evals/fixtures"


def test_v2_is_frozen_distinct_and_has_split_comments() -> None:
    v1 = json.loads((ROOT / "heldout_attacks.json").read_text(encoding="utf-8"))
    v2 = json.loads((ROOT / "heldout_attacks_v2.json").read_text(encoding="utf-8"))
    assert len(v2["cases"]) == len(v2["issues"]) == 20
    assert len({case["id"] for case in v2["cases"]}) == 20
    assert sum(len(case["attack_ids"]) > 1 for case in v2["cases"]) >= 3
    assert {row["body"] for row in v2["comments"]}.isdisjoint(
        {row["body"] for row in v1["comments"]}
    )
    categories = {case["attack_category"] for case in v2["cases"]}
    assert {"split_comments", "roleplay", "markdown_link", "portuguese"} <= categories
    comments = {row["id"]: row for row in v2["comments"]}
    for case in v2["cases"]:
        assert all(comments[attack_id]["parent_url"] == case["expected_url"]
                   for attack_id in case["attack_ids"])
