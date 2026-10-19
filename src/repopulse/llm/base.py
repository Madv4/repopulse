"""Common message and completion contract for all LLM providers."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal, cast

import httpx
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single conversation message."""

    role: Literal["system", "user", "assistant"]
    content: str


class LLMResponse(BaseModel):
    """Provider-independent text response and reported token usage."""

    text: str
    model: str
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)


class ProviderError(RuntimeError):
    """A safe error that does not expose request headers or credentials."""


class LLMProvider(ABC):
    """Replaceable synchronous completion interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the configured provider name."""

    @abstractmethod
    def complete(
        self, messages: Sequence[ChatMessage], *, max_output_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate one response from ordered conversation messages."""


def post_json(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Post JSON and return an object while keeping HTTP error details private."""
    try:
        response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as exc:
        raise ProviderError(f"LLM service returned HTTP {exc.response.status_code}") from None
    except (httpx.HTTPError, ValueError) as exc:
        raise ProviderError(f"LLM service failed: {type(exc).__name__}") from None
    if not isinstance(data, dict):
        raise ProviderError("LLM service returned an invalid JSON object")
    return cast(dict[str, Any], data)


def read_text(value: object, *, field: str) -> str:
    """Extract a required text field from a provider response."""
    if not isinstance(value, str):
        raise ProviderError(f"LLM service response missing {field}")
    return value


def read_count(value: object) -> int:
    """Normalize optional provider token counts without inventing usage."""
    return value if isinstance(value, int) and value >= 0 else 0
