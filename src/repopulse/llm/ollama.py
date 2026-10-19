"""Ollama local chat API adapter."""

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


class OllamaProvider(LLMProvider):
    """Generate with a locally running Ollama server."""

    def __init__(self, model: str, base_url: str, client: httpx.Client | None = None) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=60)

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "ollama"

    def complete(
        self, messages: Sequence[ChatMessage], *, max_output_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Request a non-streaming Ollama completion."""
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "stream": False,
        }
        if max_output_tokens is not None:
            payload["options"] = {"num_predict": max_output_tokens}
        if json_mode:
            payload["format"] = "json"
        data = post_json(
            self.client,
            f"{self.base_url}/api/chat",
            payload,
        )
        message = data.get("message")
        if not isinstance(message, dict):
            raise ProviderError("Ollama response missing message")
        return LLMResponse(
            text=read_text(message.get("content"), field="message.content"),
            model=self.model,
            prompt_tokens=read_count(data.get("prompt_eval_count")),
            completion_tokens=read_count(data.get("eval_count")),
        )
