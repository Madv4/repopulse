"""Manual, networked capture of selected public GitHub REST fields.

The committed fixture is pinned. Running this script later creates a new snapshot,
which must be reviewed and relabeled before replacing the committed one.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

API = "https://api.github.com/repos/pallets/click"
OUT = Path(__file__).resolve().parent / "fixtures/click_snapshot.json"
SELECTED = {3887, 3886, 3885, 3884, 3882, 3880, 3878, 3877, 3866, 3865, 3847, 3840}
ISSUE_FIELDS = (
    "number",
    "html_url",
    "title",
    "body",
    "user",
    "created_at",
    "updated_at",
    "closed_at",
    "labels",
    "comments",
    "pull_request",
)
PULL_FIELDS = (
    "number",
    "html_url",
    "title",
    "body",
    "user",
    "created_at",
    "updated_at",
    "closed_at",
    "merged_at",
    "labels",
    "review_comments",
)
COMMENT_FIELDS = ("html_url", "body", "user", "created_at")
COMMIT_FIELDS = ("sha", "html_url", "author", "commit")


def select(record: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    result = {key: record[key] for key in fields if key in record}
    if "user" in result and result["user"]:
        result["user"] = {"login": result["user"].get("login")}
    if "author" in result and result["author"]:
        result["author"] = {"login": result["author"].get("login")}
    if "labels" in result:
        result["labels"] = [{"name": label["name"]} for label in result["labels"]]
    if "pull_request" in result:
        result["pull_request"] = True
    if "commit" in result:
        commit = result["commit"]
        result["commit"] = {
            "message": commit.get("message"),
            "author": {"name": (commit.get("author") or {}).get("name")},
            "committer": {"date": (commit.get("committer") or {}).get("date")},
        }
    return result


def main() -> None:
    with httpx.Client(
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "RepoPulse-fixture-capture",
        },
        timeout=20,
    ) as client:

        def get(path: str, **params: str | int) -> Any:
            response = client.get(f"{API}{path}", params=params)
            response.raise_for_status()
            print(
                path,
                response.status_code,
                "remaining",
                response.headers.get("x-ratelimit-remaining"),
            )
            return response.json()

        listed = get("/issues", state="all", since="2026-09-01T00:00:00Z", per_page=100)
        issues = [select(item, ISSUE_FIELDS) for item in listed if item["number"] in SELECTED]
        assert len(issues) == len(SELECTED), {item["number"] for item in issues}
        pulls = {}
        issue_comments = {}
        review_comments = {}
        for issue in issues:
            number = issue["number"]
            if issue.get("pull_request"):
                raw_pull = get(f"/pulls/{number}")
                pulls[str(number)] = select(raw_pull, PULL_FIELDS)
                if pulls[str(number)].get("review_comments"):
                    comments = get(f"/pulls/{number}/comments", per_page=100)
                    review_comments[str(number)] = [select(c, COMMENT_FIELDS) for c in comments]
            if issue.get("comments"):
                comments = get(f"/issues/{number}/comments", per_page=100)
                issue_comments[str(number)] = [select(c, COMMENT_FIELDS) for c in comments]
        commits = get(
            "/commits", since="2026-09-01T00:00:00Z", until="2026-10-19T00:00:00Z", per_page=30
        )
        selected_commits = [select(c, COMMIT_FIELDS) for c in commits[:8]]
    snapshot = {
        "repo": "pallets/click",
        "captured_at": datetime.now(UTC).isoformat(),
        "window_start": "2026-09-01T00:00:00Z",
        "window_end": "2026-10-19T00:00:00Z",
        "source_api": API,
        "selection": (
            "Curated recent issues and PRs plus first eight commits in the window; "
            "selected fields preserve API text exactly."
        ),
        "issues": issues,
        "pulls": pulls,
        "issue_comments": issue_comments,
        "review_comments": review_comments,
        "commits": selected_commits,
    }
    OUT.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT, "bytes", OUT.stat().st_size)


if __name__ == "__main__":
    main()
