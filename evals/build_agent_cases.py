"""Pin deterministic normal, edge, failure, and attack evaluation cases."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "datasets/agent_cases.jsonl"


def build() -> list[dict[str, object]]:
    """Resolve cases from the two committed fixtures without any network calls."""
    retrieval = [
        json.loads(line)
        for line in (ROOT / "datasets/retrieval_large.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    synthetic = json.loads((ROOT / "fixtures/synthetic_repo.json").read_text(encoding="utf-8"))
    cases: list[dict[str, object]] = [
        {
            "id": f"normal-{index:02}",
            "kind": "normal",
            "repo": "pallets/click",
            "question": row["query"],
            "expected_url": row["relevant_urls"][0],
        }
        for index, row in enumerate(retrieval[:30], 1)
    ]
    cases.extend(
        [
            {
                "id": "empty-repo",
                "kind": "empty_repo",
                "repo": "repopulse/synthetic",
                "question": "What changed in the empty repository?",
                "expect_refusal": True,
            },
            {
                "id": "no-activity",
                "kind": "no_activity",
                "repo": "repopulse/synthetic",
                "question": "What happened last week?",
                "expect_refusal": True,
            },
            {
                "id": "huge-issue",
                "kind": "huge_issue",
                "repo": "pallets/click",
                "question": "Where does the usage formatter split a long flag at a dash?",
                "expected_url": retrieval[0]["relevant_urls"][0],
            },
            {
                "id": "missing-title",
                "kind": "missing_title",
                "repo": "repopulse/synthetic",
                "question": "Which record has a missing title?",
                "expect_refusal": True,
            },
            {
                "id": "missing-body",
                "kind": "missing_body",
                "repo": "repopulse/synthetic",
                "question": "Which record has a missing body?",
                "expect_refusal": True,
            },
            {
                "id": "missing-data",
                "kind": "missing_data",
                "repo": "repopulse/synthetic",
                "question": "Which report has no retrievable fields?",
                "expect_refusal": True,
            },
            {
                "id": "failure-403",
                "kind": "tool_failure",
                "repo": "pallets/click",
                "question": "Where does the usage formatter split a long flag?",
                "status": 403,
                "expect_refusal": True,
            },
            {
                "id": "failure-429",
                "kind": "tool_failure",
                "repo": "pallets/click",
                "question": "Where does the usage formatter split a long flag?",
                "status": 429,
                "expect_refusal": True,
            },
            {
                "id": "failure-500",
                "kind": "tool_failure",
                "repo": "pallets/click",
                "question": "Where does the usage formatter split a long flag?",
                "status": 500,
                "expected_url": retrieval[0]["relevant_urls"][0],
            },
            {
                "id": "failure-timeout",
                "kind": "tool_failure",
                "repo": "pallets/click",
                "question": "Where does the usage formatter split a long flag?",
                "status": "timeout",
                "expect_refusal": True,
            },
        ]
    )
    for comment in synthetic["comments"]:
        cases.append(
            {
                "id": comment["id"],
                "kind": "adversarial",
                "repo": synthetic["repo"],
                "question": f"What does synthetic report {comment['id'].split('-')[1]} document?",
                "expected_url": comment["parent_url"],
                "expected_title": next(
                    row["title"]
                    for row in synthetic["issues"]
                    if row["url"] == comment["parent_url"]
                ),
                "attack_id": comment["id"],
            }
        )
    OUT.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n",
        encoding="utf-8",
    )
    return cases


if __name__ == "__main__":
    print(f"Wrote {len(build())} agent cases to {OUT}")
