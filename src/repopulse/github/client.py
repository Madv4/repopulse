"""Read-only GitHub REST client with conditional caching and bounded retries."""

import re
import time
from collections.abc import Callable, Mapping
from email.utils import parsedate_to_datetime
from typing import Any, cast

import httpx

from repopulse.config import Settings
from repopulse.github.cache import DiskCache
from repopulse.github.models import (
    CacheEntry,
    GitHubAPIError,
    GitHubNotFoundError,
    GitHubRateLimitError,
    GitHubResponse,
    JsonPayload,
)

_NEXT_LINK = re.compile(r'<([^>]+)>;\s*rel="next"')
_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class GitHubClient:
    """Fetch public GitHub data, optionally with a PAT for a higher limit."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.Client | None = None,
        base_url: str = "https://api.github.com",
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.base_url = base_url.rstrip("/")
        self._origin = httpx.URL(self.base_url)
        if self._origin.scheme != "https" and self._origin.host not in {"localhost", "127.0.0.1"}:
            raise ValueError("GitHub API base URL must use HTTPS")
        self._token = settings.github_token.get_secret_value() if settings.github_token else None
        self._cache = DiskCache(settings.github_cache_dir, self._token)
        self._client = client or httpx.Client(timeout=settings.github_timeout_seconds)
        self._clock = clock
        self._sleep = sleep

    def _url(self, path_or_url: str, params: Mapping[str, str | int] | None = None) -> str:
        candidate = httpx.URL(path_or_url)
        if candidate.is_absolute_url:
            if (
                candidate.scheme != self._origin.scheme
                or candidate.host != self._origin.host
                or candidate.port != self._origin.port
            ):
                raise ValueError("Pagination URL must stay on the GitHub API origin")
            if params:
                raise ValueError("Query parameters cannot be added to a pagination URL")
            return str(candidate)
        if not path_or_url.startswith("/") or path_or_url.startswith("//"):
            raise ValueError("GitHub API path must start with one slash")
        if ".." in path_or_url.split("/") or "#" in path_or_url:
            raise ValueError("GitHub API path contains an unsupported segment")
        url = httpx.URL(self.base_url + path_or_url)
        if params:
            url = url.copy_merge_params(dict(sorted(params.items())))
        return str(url)

    @staticmethod
    def _payload(response: httpx.Response) -> JsonPayload:
        try:
            data = response.json()
        except ValueError:
            raise GitHubAPIError("GitHub returned invalid JSON") from None
        if not isinstance(data, (dict, list)) or (
            isinstance(data, list) and not all(isinstance(item, dict) for item in data)
        ):
            raise GitHubAPIError("GitHub returned an unexpected JSON shape")
        return cast(JsonPayload, data)

    def _retry_after(self, headers: Mapping[str, str]) -> float | None:
        value = headers.get("retry-after")
        if value:
            try:
                return max(0.0, float(value))
            except ValueError:
                try:
                    return max(0.0, parsedate_to_datetime(value).timestamp() - self._clock())
                except (TypeError, ValueError):
                    pass
        reset = headers.get("x-ratelimit-reset")
        if reset:
            try:
                return max(0.0, float(reset) - self._clock())
            except ValueError:
                pass
        return None

    def get(
        self, path_or_url: str, params: Mapping[str, str | int] | None = None
    ) -> GitHubResponse:
        """GET once, using a fresh cache entry or revalidating a stale one."""
        url = self._url(path_or_url, params)
        cached = self._cache.read(url)
        if cached and self._clock() - cached.fetched_at < self.settings.github_cache_ttl_seconds:
            return GitHubResponse(data=cached.data, link=cached.link, from_cache=True)
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "RepoPulse/0.2",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        if cached and cached.etag:
            headers["If-None-Match"] = cached.etag
        if cached and cached.last_modified:
            headers["If-Modified-Since"] = cached.last_modified
        for attempt in range(self.settings.github_max_retries + 1):
            try:
                response = self._client.get(url, headers=headers)
            except httpx.TransportError as exc:
                if attempt < self.settings.github_max_retries:
                    self._sleep(2**attempt)
                    continue
                error_type = type(exc).__name__
                raise GitHubAPIError(f"GitHub network request failed: {error_type}") from None
            if response.status_code == 304:
                if cached is None:
                    raise GitHubAPIError("GitHub returned 304 without a cached response")
                cached.fetched_at = self._clock()
                self._cache.write(url, cached)
                return GitHubResponse(data=cached.data, link=cached.link, from_cache=True)
            if (
                response.status_code == 429
                or response.status_code == 403
                and (
                    response.headers.get("x-ratelimit-remaining") == "0"
                    or "retry-after" in response.headers
                    or "rate limit" in response.text.lower()
                    or "abuse" in response.text.lower()
                )
            ):
                raise GitHubRateLimitError(self._retry_after(response.headers))
            if response.status_code == 404:
                raise GitHubNotFoundError("GitHub resource not found or inaccessible")
            if response.status_code >= 500 and attempt < self.settings.github_max_retries:
                self._sleep(2**attempt)
                continue
            if response.status_code >= 400:
                raise GitHubAPIError(f"GitHub returned HTTP {response.status_code}")
            data = self._payload(response)
            entry = CacheEntry(
                fetched_at=self._clock(),
                etag=response.headers.get("etag"),
                last_modified=response.headers.get("last-modified"),
                link=response.headers.get("link"),
                data=data,
            )
            self._cache.write(url, entry)
            return GitHubResponse(data=data, link=entry.link)
        raise GitHubAPIError("GitHub request exhausted retries")

    def get_paginated(
        self,
        path: str,
        params: Mapping[str, str | int] | None = None,
        *,
        max_pages: int = 20,
    ) -> list[dict[str, Any]]:
        """Follow only GitHub's next links, with a page cap and loop detection."""
        if max_pages < 1:
            raise ValueError("max_pages must be at least 1")
        url = self._url(path, params)
        seen: set[str] = set()
        items: list[dict[str, Any]] = []
        for _ in range(max_pages):
            if url in seen:
                raise GitHubAPIError("GitHub pagination loop detected")
            seen.add(url)
            page = self.get(url)
            if not isinstance(page.data, list):
                raise GitHubAPIError("GitHub pagination expected a JSON array")
            items.extend(page.data)
            match = _NEXT_LINK.search(page.link or "")
            if not match:
                return items
            url = self._url(match.group(1))
        raise GitHubAPIError(f"GitHub pagination exceeded max_pages={max_pages}")

    @staticmethod
    def repo_path(repo: str) -> str:
        """Validate the owner/name path before composing repository endpoints."""
        if not _REPO.fullmatch(repo) or any(part in {".", ".."} for part in repo.split("/")):
            raise ValueError("repo must have the form owner/name")
        return f"/repos/{repo}"

    def list_issues(self, repo: str, *, state: str = "all") -> list[dict[str, Any]]:
        """List repository issues (GitHub also includes PRs in this endpoint)."""
        return self.get_paginated(
            f"{self.repo_path(repo)}/issues", {"state": state, "per_page": 100}
        )

    def search_issues(self, repo: str, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        """Search issues in one repository, excluding pull requests."""
        self.repo_path(repo)
        if not query.strip():
            raise ValueError("search query must not be empty")
        response = self.get(
            "/search/issues",
            {"q": f"repo:{repo} is:issue {query}", "per_page": min(max(limit, 1), 20)},
        )
        if not isinstance(response.data, dict) or not isinstance(response.data.get("items"), list):
            raise GitHubAPIError("GitHub issue search returned an unexpected shape")
        return cast(list[dict[str, Any]], response.data["items"])[:limit]

    def get_issue(self, repo: str, number: int) -> GitHubResponse:
        """Fetch one issue."""
        return self.get(f"{self.repo_path(repo)}/issues/{number}")

    def get_pull_request(self, repo: str, number: int) -> GitHubResponse:
        """Fetch one pull request."""
        return self.get(f"{self.repo_path(repo)}/pulls/{number}")

    def get_pr_files(self, repo: str, number: int, *, limit: int = 10) -> list[dict[str, Any]]:
        """Fetch changed-file metadata for a pull request."""
        response = self.get(
            f"{self.repo_path(repo)}/pulls/{number}/files", {"per_page": min(max(limit, 1), 100)}
        )
        if not isinstance(response.data, list):
            raise GitHubAPIError("GitHub PR files returned an unexpected shape")
        return response.data[:limit]

    def list_commits(self, repo: str, *, limit: int = 5) -> list[dict[str, Any]]:
        """List repository commits."""
        response = self.get(
            f"{self.repo_path(repo)}/commits", {"per_page": min(max(limit, 1), 100)}
        )
        if not isinstance(response.data, list):
            raise GitHubAPIError("GitHub commits returned an unexpected shape")
        return response.data[:limit]

    def list_milestones(self, repo: str, *, limit: int = 5) -> list[dict[str, Any]]:
        """List repository milestones."""
        response = self.get(
            f"{self.repo_path(repo)}/milestones",
            {"state": "all", "per_page": min(max(limit, 1), 100)},
        )
        if not isinstance(response.data, list):
            raise GitHubAPIError("GitHub milestones returned an unexpected shape")
        return response.data[:limit]

    def list_releases(self, repo: str, *, limit: int = 5) -> list[dict[str, Any]]:
        """List repository releases."""
        response = self.get(
            f"{self.repo_path(repo)}/releases", {"per_page": min(max(limit, 1), 100)}
        )
        if not isinstance(response.data, list):
            raise GitHubAPIError("GitHub releases returned an unexpected shape")
        return response.data[:limit]

    def list_issues_since(self, repo: str, since: str) -> list[dict[str, Any]]:
        """List issues and PR summaries updated since an ISO 8601 timestamp."""
        return self.get_paginated(
            f"{self.repo_path(repo)}/issues",
            {
                "state": "all",
                "since": since,
                "sort": "updated",
                "direction": "asc",
                "per_page": 100,
            },
        )

    def list_issue_comments(self, repo: str, number: int) -> list[dict[str, Any]]:
        """List issue comments, which also covers general PR discussion."""
        return self.get_paginated(
            f"{self.repo_path(repo)}/issues/{number}/comments", {"per_page": 100}
        )

    def list_pr_review_comments(self, repo: str, number: int) -> list[dict[str, Any]]:
        """List line-level review comments on a pull request."""
        return self.get_paginated(
            f"{self.repo_path(repo)}/pulls/{number}/comments", {"per_page": 100}
        )

    def list_commits_between(self, repo: str, since: str, until: str) -> list[dict[str, Any]]:
        """List commits whose GitHub timestamp lies in the requested window."""
        return self.get_paginated(
            f"{self.repo_path(repo)}/commits",
            {"since": since, "until": until, "per_page": 100},
        )
