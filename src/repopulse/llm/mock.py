"""Deterministic, scriptable provider for offline tests and CI."""

from collections.abc import Callable, Iterable, Sequence

from repopulse.llm.base import ChatMessage, LLMProvider, LLMResponse, ProviderError


class MockProvider(LLMProvider):
    """Return scripted responses in order, or a fixed fallback response."""

    def __init__(
        self,
        responses: Iterable[str | LLMResponse] | None = None,
        *,
        fallback: str = "Mock response",
        responder: Callable[[tuple[ChatMessage, ...]], str] | None = None,
    ) -> None:
        self._responses = iter(responses) if responses is not None else None
        self._fallback = fallback
        self._responder = responder
        self.calls: list[tuple[ChatMessage, ...]] = []

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "mock"

    def complete(
        self, messages: Sequence[ChatMessage], *, max_output_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Return a deterministic scripted completion without network access."""
        call = tuple(messages)
        self.calls.append(call)
        text: str | LLMResponse
        if self._responder is not None:
            text = self._responder(call)
        elif self._responses is not None:
            try:
                text = next(self._responses)
            except StopIteration as exc:
                raise ProviderError("Mock response script exhausted") from exc
        else:
            text = self._fallback
        return text if isinstance(text, LLMResponse) else LLMResponse(text=text, model="mock")
