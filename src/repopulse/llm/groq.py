"""Groq's OpenAI-compatible chat completions adapter."""

from collections.abc import Sequence

import httpx

from repopulse.llm.base import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    ProviderError,
    post_json,
    read_count,
    read_text,
)


class GroqProvider(LLMProvider):
    """Generate through Groq's free-tier compatible API."""

    def __init__(self, model: str, api_key: str, client: httpx.Client | None = None) -> None:
        if not api_key:
            raise ProviderError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        self.model = model
        self._api_key = api_key
        self.client = client or httpx.Client(timeout=60)

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "groq"

    def complete(
        self, messages: Sequence[ChatMessage], *, max_output_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Request a Groq chat completion."""
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
        }
        if max_output_tokens is not None:
            payload["max_completion_tokens"] = max_output_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = post_json(
            self.client,
            "https://api.groq.com/openai/v1/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ProviderError("Groq response missing choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise ProviderError("Groq response missing message")
        usage = data.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        return LLMResponse(
            text=read_text(message.get("content"), field="message.content"),
            model=self.model,
            prompt_tokens=read_count(usage.get("prompt_tokens")),
            completion_tokens=read_count(usage.get("completion_tokens")),
        )
