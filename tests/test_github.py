"""Offline tests for GitHub caching, pagination and failures."""

from pathlib import Path

import httpx
import pytest

from repopulse.config import Settings
from repopulse.github import GitHubClient, GitHubNotFoundError, GitHubRateLimitError
from repopulse.github.models import GitHubAPIError


def settings(path: Path, **kwargs: object) -> Settings:
    return Settings(_env_file=None, github_cache_dir=path, **kwargs)


def test_fresh_cache_avoids_second_request(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"number": 1}, headers={"ETag": '"abc"'})

    client = GitHubClient(
        settings(tmp_path),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        clock=lambda: 1000,
    )
    assert client.get("/repos/a/b/issues/1").data == {"number": 1}
    assert client.get("/repos/a/b/issues/1").from_cache
    assert len(requests) == 1
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_stale_etag_revalidation(tmp_path: Path) -> None:
    headers_seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        headers_seen.append(request.headers.get("if-none-match"))
        if len(headers_seen) == 1:
            return httpx.Response(200, json={"number": 2}, headers={"ETag": '"v1"'})
        return httpx.Response(304)

    now = [1000.0]
    client = GitHubClient(
        settings(tmp_path, github_cache_ttl_seconds=10),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        clock=lambda: now[0],
    )
    client.get("/repos/a/b/issues/2")
    now[0] = 1011
    result = client.get("/repos/a/b/issues/2")
    assert result.from_cache and result.data == {"number": 2}
    assert headers_seen == [None, '"v1"']
    now[0] = 1012
    assert client.get("/repos/a/b/issues/2").from_cache
    assert len(headers_seen) == 2


def test_pagination_follows_next_link_and_caches_pages(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.params.get("page") == "2":
            return httpx.Response(200, json=[{"number": 2}])
        next_url = str(request.url.copy_set_param("page", "2"))
        return httpx.Response(
            200, json=[{"number": 1}], headers={"Link": f'<{next_url}>; rel="next"'}
        )

    client = GitHubClient(
        settings(tmp_path), client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    assert [item["number"] for item in client.list_issues("a/b")] == [1, 2]
    assert [item["number"] for item in client.list_issues("a/b")] == [1, 2]
    assert len(calls) == 2
    with pytest.raises(GitHubAPIError, match="max_pages"):
        client.get_paginated("/repos/a/b/issues", max_pages=1)


def test_issue_search_and_commit_list_use_bounded_single_pages(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/search/issues":
            assert request.url.params["q"] == "repo:a/b is:issue colors"
            assert request.url.params["per_page"] == "2"
            return httpx.Response(200, json={"items": [{"number": 1}, {"number": 2}]})
        assert request.url.path == "/repos/a/b/commits"
        assert request.url.params["per_page"] == "1"
        return httpx.Response(200, json=[{"sha": "abc"}], headers={
            "Link": '<https://api.github.com/repos/a/b/commits?page=2>; rel="next"'
        })

    client = GitHubClient(
        settings(tmp_path), client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    assert len(client.search_issues("a/b", "colors", limit=2)) == 2
    assert client.list_commits("a/b", limit=1) == [{"sha": "abc"}]
    assert len(calls) == 2


def test_rate_limit_is_not_retried(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            403, headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1060"}
        )

    client = GitHubClient(
        settings(tmp_path),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        clock=lambda: 1000,
    )
    with pytest.raises(GitHubRateLimitError) as exc:
        client.get("/repos/a/b/issues")
    assert exc.value.retry_after_seconds == 60
    assert calls == 1


def test_server_error_retries_then_succeeds(tmp_path: Path) -> None:
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500 if calls < 3 else 200, json={"ok": True})

    client = GitHubClient(
        settings(tmp_path),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=sleeps.append,
    )
    assert client.get("/repos/a/b").data == {"ok": True}
    assert calls == 3 and sleeps == [1, 2]


def test_not_found_and_invalid_paths(tmp_path: Path) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(404))
    client = GitHubClient(settings(tmp_path), client=httpx.Client(transport=transport))
    with pytest.raises(GitHubNotFoundError):
        client.get_issue("a/b", 10)
    for path in ("https://evil.example/repos/a/b", "//evil.example/path", "/repos/../bad"):
        with pytest.raises(ValueError):
            client.get(path)
    with pytest.raises(ValueError):
        client.repo_path("a/../b")


def test_token_scopes_cache_without_storing_token(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    path = "/repos/a/b"
    public = GitHubClient(settings(tmp_path), client=httpx.Client(transport=transport))
    private = GitHubClient(
        settings(tmp_path, github_token="top-secret"), client=httpx.Client(transport=transport)
    )
    public.get(path)
    private.get(path)
    assert calls == 2
    assert len(list(tmp_path.glob("*.json"))) == 2
    assert all("top-secret" not in file.read_text() for file in tmp_path.glob("*.json"))
