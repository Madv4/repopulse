"""Record real Click discussion comments for a benign false-positive audit.

This is an explicit network capture, never called by tests or offline eval.
The 400-item issue/PR fixture is read unchanged.
"""

import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from repopulse.config import Settings
from repopulse.github.client import GitHubClient

ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "fixtures/click_large_snapshot.json"
OUT = ROOT / "fixtures/click_benign_comments.json"


def capture() -> dict[str, Any]:
    """Select 200 nonempty real comments across the pinned issue/PR cohort."""
    source_bytes = PARENT.read_bytes()
    source = json.loads(source_bytes)
    parents = {int(row["number"]): row for row in source["issues"]}
    cutoff = datetime.fromisoformat(source["captured_at"])
    github = GitHubClient(Settings())
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    pages = 0
    for page in range(1, 11):
        response = github.get(
            "/repos/pallets/click/issues/comments",
            {"per_page": 100, "sort": "updated", "direction": "desc", "page": page},
        )
        if not isinstance(response.data, list):
            raise ValueError("GitHub comments endpoint returned a non-list response")
        pages += 1
        for row in response.data:
            number = int(str(row["issue_url"]).rsplit("/", 1)[-1])
            created = datetime.fromisoformat(str(row["created_at"]).replace("Z", "+00:00"))
            body = str(row.get("body") or "")
            if number not in parents or created > cutoff or not body.strip():
                continue
            grouped[number].append({
                "id": int(row["id"]),
                "url": str(row["html_url"]),
                "issue_number": number,
                "issue_url": str(parents[number]["html_url"]),
                "issue_title": str(parents[number]["title"]),
                "kind": "pull_request" if parents[number].get("pull_request") else "issue",
                "author": str((row.get("user") or {}).get("login") or "unknown"),
                "created_at": str(row["created_at"]),
                "body": body,
            })
        if sum(map(len, grouped.values())) >= 200 or len(response.data) < 100:
            break
    if sum(map(len, grouped.values())) < 200:
        raise ValueError("Fewer than 200 qualifying real comments were available")
    selected: list[dict[str, Any]] = []
    for comments in grouped.values():
        comments.sort(key=lambda row: row["id"], reverse=True)
    while len(selected) < 200:
        for number in sorted(grouped):
            if grouped[number]:
                selected.append(grouped[number].pop(0))
                if len(selected) == 200:
                    break
    result = {
        "repo": source["repo"],
        "captured_at": datetime.now(UTC).isoformat(),
        "parent_snapshot_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "parent_snapshot_captured_at": source["captured_at"],
        "source_api": "https://api.github.com/repos/pallets/click/issues/comments",
        "selection": (
            "Newest updated public issue/PR comments created by the parent snapshot cutoff, "
            "restricted to its 400 issue/PR numbers; round-robin by parent number and descending "
            "comment ID to select 200 nonempty comments. No planted attack fixtures."
        ),
        "pages_fetched": pages,
        "comments": selected,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    recorded = capture()
    print(f"Recorded {len(recorded['comments'])} comments in {OUT}")
