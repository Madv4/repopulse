"""Normalized evidence records shared by ingestion and retrieval."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

DocumentKind = Literal["issue", "pull_request", "commit"]


class SourceComment(BaseModel):
    """Issue or pull request comment from GitHub."""

    url: str
    body: str
    author: str
    created_at: datetime


class SourceDocument(BaseModel):
    """One issue, pull request, or commit and its citation metadata."""

    repo: str
    kind: DocumentKind
    source_id: str
    number: int | None = None
    url: str
    title: str
    body: str = ""
    author: str = ""
    created_at: datetime
    updated_at: datetime | None = None
    labels: list[str] = Field(default_factory=list)
    comments: list[SourceComment] = Field(default_factory=list)

    def render_text(self) -> str:
        """Build searchable text while retaining comment boundaries and authors."""
        heading = f"{self.kind.replace('_', ' ').title()}: {self.title}"
        parts = [heading, self.body.strip()]
        parts.extend(
            f"Comment by {comment.author} ({comment.url}):\n{comment.body.strip()}"
            for comment in self.comments
            if comment.body.strip()
        )
        return "\n\n".join(part for part in parts if part)


class Chunk(BaseModel):
    """A retrievable passage with enough metadata for an evidence citation."""

    id: str
    source_id: str
    repo: str
    kind: DocumentKind
    number: int | None
    url: str
    author: str
    created_at: datetime
    updated_at: datetime | None
    labels: list[str]
    strategy: Literal["fixed", "structure"]
    index: int = Field(ge=0)
    text: str = Field(min_length=1)
