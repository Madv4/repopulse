"""Typed GitHub responses and safe API exceptions."""

from typing import Any, TypeAlias

from pydantic import BaseModel, Field

JsonObject: TypeAlias = dict[str, Any]
JsonPayload: TypeAlias = JsonObject | list[JsonObject]


class GitHubResponse(BaseModel):
    """One GET response, including pagination metadata."""

    data: JsonPayload
    link: str | None = None
    from_cache: bool = False


class CacheEntry(BaseModel):
    """Persisted response body and conditional-request validators."""

    fetched_at: float = Field(ge=0)
    etag: str | None = None
    last_modified: str | None = None
    link: str | None = None
    data: JsonPayload


class GitHubAPIError(RuntimeError):
    """GitHub request failed; no URL query or token appears in the message."""


class GitHubNotFoundError(GitHubAPIError):
    """Requested GitHub resource was not found or is not accessible."""


class GitHubRateLimitError(GitHubAPIError):
    """Rate limiting requires the caller to wait before another request."""

    def __init__(self, retry_after_seconds: float | None) -> None:
        self.retry_after_seconds = retry_after_seconds
        detail = (
            f"; retry after about {retry_after_seconds:.0f} seconds"
            if retry_after_seconds is not None
            else "; check GitHub rate-limit headers before retrying"
        )
        super().__init__(f"GitHub rate limit reached{detail}")
