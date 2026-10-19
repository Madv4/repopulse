"""In-process conversation context, never persisted to SQLite."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from repopulse.agent.graph import Answer


class SessionTurn(BaseModel):
    """A user question and cited URLs, without untrusted source text."""

    question: str
    cited_urls: list[str] = Field(default_factory=list)


class SessionMemory:
    """Keep a bounded window of prior turns for one live process."""

    def __init__(self, max_turns: int = 6) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self._turns: deque[SessionTurn] = deque(maxlen=max_turns)

    def remember(self, question: str, answer: Answer) -> None:
        """Store the user's wording and citation URLs, never retrieved passages."""
        claims = [answer.summary, *answer.key_changes, *answer.risks]
        urls = list(dict.fromkeys(url for claim in claims if claim for url in claim.citations))
        self._turns.append(SessionTurn(question=question, cited_urls=urls))

    def recent(self) -> list[SessionTurn]:
        """Return a detached copy of the bounded conversation history."""
        return [turn.model_copy(deep=True) for turn in self._turns]
