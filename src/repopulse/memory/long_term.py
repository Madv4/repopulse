"""Durable, policy-gated SQLite memory with no automatic retrieval writes."""

import json
import re
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

from repopulse.agent.citations import supports_text
from repopulse.mcp_server import Evidence
from repopulse.memory.policy import MemoryDraft, MemoryKind, MemoryPolicy, MemorySource


class MemoryRecord(BaseModel):
    """One durable fact, preference, or explicitly promoted summary."""

    id: str
    text: str
    kind: MemoryKind
    source: MemorySource
    repo: str | None
    confidence: float
    created_at: datetime
    expires_at: datetime | None
    source_urls: list[str]


class MemoryStore:
    """Persist approved memory records in a local SQLite database."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    source TEXT NOT NULL,
                    repo TEXT,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    source_urls TEXT NOT NULL
                )"""
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS memories_scope ON memories(repo, expires_at)"
            )

    def _insert(self, draft: MemoryDraft, *, verified: bool = False) -> MemoryRecord:
        MemoryPolicy.validate(draft, verified=verified)
        record = MemoryRecord(
            id=uuid4().hex,
            text=draft.text.strip(),
            kind=draft.kind,
            source=draft.source,
            repo=draft.repo,
            confidence=draft.confidence,
            created_at=datetime.now(UTC),
            expires_at=draft.expires_at.astimezone(UTC) if draft.expires_at else None,
            source_urls=draft.source_urls,
        )
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(
                """INSERT INTO memories
                   (id, text, kind, source, repo, confidence, created_at, expires_at, source_urls)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.id,
                    record.text,
                    record.kind,
                    record.source,
                    record.repo,
                    record.confidence,
                    record.created_at.isoformat(),
                    record.expires_at.isoformat() if record.expires_at else None,
                    json.dumps(record.source_urls),
                ),
            )
        return record

    def add_user(
        self,
        text: str,
        *,
        kind: MemoryKind = "preference",
        repo: str | None = None,
        confidence: float = 1.0,
        expires_at: datetime | None = None,
    ) -> MemoryRecord:
        """Save an explicit user-stated durable fact or preference."""
        return self._insert(
            MemoryDraft(
                text=text,
                kind=kind,
                source="user",
                repo=repo,
                confidence=confidence,
                expires_at=expires_at,
            )
        )

    def add_verified_summary(
        self,
        text: str,
        citations: list[str],
        evidence: list[Evidence],
        *,
        repo: str | None = None,
        confidence: float = 1.0,
        expires_at: datetime | None = None,
    ) -> MemoryRecord:
        """Explicitly promote an extractive summary after checking every citation."""
        urls = list(dict.fromkeys(citations))
        if repo is not None and any(
            not url.startswith(f"https://github.com/{repo}/") for url in urls
        ):
            raise ValueError("summary citation is outside the scoped repository")
        if not urls or any(
            not any(item.url == url and supports_text(text, item) for item in evidence)
            for url in urls
        ):
            raise ValueError("summary is not supported by every cited source")
        return self._insert(
            MemoryDraft(
                text=text,
                kind="summary",
                source="agent_verified",
                repo=repo,
                confidence=confidence,
                expires_at=expires_at,
                source_urls=urls,
            ),
            verified=True,
        )

    def list_active(self, *, repo: str | None = None) -> list[MemoryRecord]:
        """List active records; a repo view also includes global memories."""
        if repo is not None:
            from repopulse.github.client import GitHubClient

            GitHubClient.repo_path(repo)
        query = (
            "SELECT id, text, kind, source, repo, confidence, created_at, expires_at, "
            "source_urls FROM memories WHERE (expires_at IS NULL OR expires_at > ?)"
        )
        parameters: list[str] = [datetime.now(UTC).isoformat()]
        if repo is not None:
            query += " AND (repo IS NULL OR repo = ?)"
            parameters.append(repo)
        query += " ORDER BY created_at DESC, id DESC"
        with closing(sqlite3.connect(self.path)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(query, parameters).fetchall()
        return [
            MemoryRecord(
                id=row["id"],
                text=row["text"],
                kind=row["kind"],
                source=row["source"],
                repo=row["repo"],
                confidence=row["confidence"],
                created_at=datetime.fromisoformat(row["created_at"]),
                expires_at=(
                    datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None
                ),
                source_urls=json.loads(row["source_urls"]),
            )
            for row in rows
        ]

    def relevant(self, question: str, repo: str, *, limit: int = 5) -> list[MemoryRecord]:
        """Rank active scoped memories for planning without an external service."""
        if limit < 1:
            raise ValueError("limit must be positive")
        terms = set(re.findall(r"[a-z0-9]{3,}", question.casefold()))
        candidates = self.list_active(repo=repo)

        def score(record: MemoryRecord) -> tuple[int, datetime]:
            overlap = len(terms.intersection(re.findall(r"[a-z0-9]{3,}", record.text.casefold())))
            weight = overlap * 3 + (2 if record.repo == repo else 0)
            if record.kind == "preference":
                weight += 1
            return weight, record.created_at

        ranked = sorted(candidates, key=score, reverse=True)
        return [record for record in ranked if score(record)[0] > 0][:limit]

    def forget(self, memory_id: str) -> bool:
        """Delete one record by its opaque ID."""
        with closing(sqlite3.connect(self.path)) as connection, connection:
            result = connection.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            return result.rowcount > 0
