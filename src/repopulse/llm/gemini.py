"""Gemini generateContent REST adapter."""

from collections.abc import Sequence
from urllib.parse import quote

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


class GeminiProvider(LLMProvider):
    """Generate through the Google AI Studio Gemini API."""

    def __init__(self, model: str, api_key: str, client: httpx.Client | None = None) -> None:
        if not api_key:
            raise ProviderError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
        self.model = model
        self._api_key = api_key
        self.client = client or httpx.Client(timeout=60)

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "gemini"

    def complete(
        self, messages: Sequence[ChatMessage], *, max_output_tokens: int | None = None
    ) -> LLMResponse:
        """Request Gemini text generation with system instructions separated."""
        system = "\n\n".join(message.content for message in messages if message.role == "system")
        contents = [
            {
                "role": "model" if message.role == "assistant" else "user",
                "parts": [{"text": message.content}],
            }
            for message in messages
            if message.role != "system"
        ]
        if not contents:
            raise ProviderError("Gemini requires at least one non-system message")
        payload: dict[str, object] = {"contents": contents}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if max_output_tokens is not None:
            payload["generationConfig"] = {"maxOutputTokens": max_output_tokens}
        data = post_json(
            self.client,
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{quote(self.model, safe='')}:generateContent",
            payload,
            headers={"x-goog-api-key": self._api_key},
        )
        candidates = data.get("candidates")
        if (
            not isinstance(candidates, list)
            or not candidates
            or not isinstance(candidates[0], dict)
        ):
            raise ProviderError("Gemini response missing candidates")
        content = candidates[0].get("content")
        if not isinstance(content, dict) or not isinstance(content.get("parts"), list):
            raise ProviderError("Gemini response missing content parts")
        parts = content["parts"]
        text = "".join(
            read_text(part.get("text"), field="part.text")
            for part in parts
            if isinstance(part, dict) and "text" in part
        )
        if not text:
            raise ProviderError("Gemini response contains no text")
        usage = data.get("usageMetadata")
        usage = usage if isinstance(usage, dict) else {}
        return LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=read_count(usage.get("promptTokenCount")),
            completion_tokens=read_count(usage.get("candidatesTokenCount")),
        )
