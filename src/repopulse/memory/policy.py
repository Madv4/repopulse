"""Trusted provenance and durable-value rules for long-term memory."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from repopulse.github.client import GitHubClient

MemoryKind = Literal["preference", "fact", "summary"]
MemorySource = Literal["user", "agent_verified", "github"]


class MemoryDraft(BaseModel):
    """Candidate memory with explicit provenance and expiry."""

    text: str = Field(min_length=12, max_length=500)
    kind: MemoryKind
    source: MemorySource
    repo: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    expires_at: datetime | None = None
    source_urls: list[str] = Field(default_factory=list)


class MemoryPolicy:
    """Accept explicit user memories or explicitly promoted verified summaries."""

    @staticmethod
    def validate(draft: MemoryDraft, *, verified: bool = False) -> MemoryDraft:
        """Reject untrusted, trivial, expired, or unverified content."""
        if draft.source == "github":
            raise ValueError("GitHub content cannot be saved to long-term memory")
        if not draft.text.strip() or len(draft.text.strip()) < 12:
            raise ValueError("memory must be a durable, nontrivial statement")
        if draft.repo is not None:
            GitHubClient.repo_path(draft.repo)
        if draft.expires_at is not None:
            if draft.expires_at.tzinfo is None:
                raise ValueError("memory expiry must include a timezone")
            if draft.expires_at <= datetime.now(UTC):
                raise ValueError("memory expiry must be in the future")
        if draft.source == "agent_verified":
            if not verified or draft.kind != "summary" or not draft.source_urls:
                raise ValueError("agent summaries require verified evidence and explicit promotion")
        elif draft.kind == "summary" or draft.source_urls:
            raise ValueError("user-stated memory must be a fact or preference without source URLs")
        return draft
