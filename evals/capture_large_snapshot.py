"""Explicit network capture of a large public GitHub issue/PR snapshot.

This script is never called by tests or `make eval`. Re-capture requires label review.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from repopulse.config import Settings

ROOT = Path(__file__).resolve().parent
REPO = "pallets/click"
OUT = ROOT / "fixtures/click_large_snapshot.json"


def capture() -> dict[str, Any]:
    """Fetch four 100-record pages from GitHub's public issue listing."""
    settings = Settings()
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "RepoPulse-eval"}
    if settings.github_token is not None:
        headers["Authorization"] = f"Bearer {settings.github_token.get_secret_value()}"
    rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=30, headers=headers) as client:
        for page in range(1, 5):
            response = client.get(
                f"https://api.github.com/repos/{REPO}/issues",
                params={
                    "state": "all",
                    "sort": "updated",
                    "direction": "desc",
                    "per_page": 100,
                    "page": page,
                },
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, list):
                raise ValueError("GitHub did not return a list")
            for item in data:
                rows.append(
                    {
                        "number": item["number"],
                        "html_url": item["html_url"],
                        "title": item.get("title") or "",
                        "body": item.get("body") or "",
                        "created_at": item["created_at"],
                        "updated_at": item["updated_at"],
                        "closed_at": item.get("closed_at"),
                        "user": {"login": (item.get("user") or {}).get("login", "unknown")},
                        "labels": [label.get("name", "") for label in item.get("labels", [])],
                        "comments": item.get("comments", 0),
                        "pull_request": item.get("pull_request"),
                    }
                )
    if len(rows) < 300 or len({row["html_url"] for row in rows}) != len(rows):
        raise ValueError("Capture lacks 300 unique issues/PRs")
    snapshot = {
        "repo": REPO,
        "captured_at": datetime.now(UTC).isoformat(),
        "source_api": f"https://api.github.com/repos/{REPO}/issues",
        "selection": "Four pages of 100 issues/PRs, state=all, sort=updated desc",
        "issues": rows,
    }
    OUT.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return snapshot


if __name__ == "__main__":
    recorded = capture()
    print(f"Captured {len(recorded['issues'])} real issue/PR records in {OUT}")
