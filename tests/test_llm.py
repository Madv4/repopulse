"""Provider request/response contract tests using local mock HTTP."""

import json

import httpx
import pytest

from repopulse.llm import ChatMessage, ProviderError
from repopulse.llm.gemini import GeminiProvider
from repopulse.llm.groq import GroqProvider
from repopulse.llm.mock import MockProvider
from repopulse.llm.ollama import OllamaProvider

MESSAGES = [
    ChatMessage(role="system", content="Be concise"),
    ChatMessage(role="user", content="Hi"),
]


def test_mock_script_and_call_capture() -> None:
    provider = MockProvider(["one", "two"])
    assert provider.complete(MESSAGES).text == "one"
    assert provider.complete(MESSAGES).text == "two"
    assert len(provider.calls) == 2
    with pytest.raises(ProviderError, match="exhausted"):
        provider.complete(MESSAGES)
    responder = MockProvider(responder=lambda call: call[-1].content.upper())
    assert responder.complete(MESSAGES).text == "HI"


def test_ollama_request_and_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = json.loads(request.content)
        assert body["stream"] is False
        assert body["messages"][1] == {"role": "user", "content": "Hi"}
        return httpx.Response(
            200, json={"message": {"content": "hello"}, "prompt_eval_count": 2, "eval_count": 3}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = OllamaProvider("test", "http://127.0.0.1:11434/", client).complete(MESSAGES)
    assert (result.text, result.prompt_tokens, result.completion_tokens) == ("hello", 2, 3)


def test_groq_request_and_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/openai/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer secret"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 5},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = GroqProvider("test", "secret", client).complete(MESSAGES)
    assert (result.text, result.prompt_tokens, result.completion_tokens) == ("hello", 4, 5)


def test_gemini_request_and_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/models/test:generateContent")
        assert request.headers["x-goog-api-key"] == "secret"
        body = json.loads(request.content)
        assert body["systemInstruction"]["parts"][0]["text"] == "Be concise"
        assert body["contents"] == [{"role": "user", "parts": [{"text": "Hi"}]}]
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "hel"}, {"text": "lo"}]}}],
                "usageMetadata": {"promptTokenCount": 6, "candidatesTokenCount": 7},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = GeminiProvider("test", "secret", client).complete(MESSAGES)
    assert (result.text, result.prompt_tokens, result.completion_tokens) == ("hello", 6, 7)


@pytest.mark.parametrize("provider", ["ollama", "groq", "gemini"])
def test_http_error_is_sanitized(provider: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "secret token appeared here"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    selected = {
        "ollama": OllamaProvider("test", "http://localhost:11434", client),
        "groq": GroqProvider("test", "secret", client),
        "gemini": GeminiProvider("test", "secret", client),
    }[provider]
    with pytest.raises(ProviderError) as exc:
        selected.complete(MESSAGES)
    assert "401" in str(exc.value)
    assert "secret" not in str(exc.value)


def test_invalid_provider_response_is_clear() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={}))
    client = httpx.Client(transport=transport)
    with pytest.raises(ProviderError, match="missing message"):
        OllamaProvider("test", "http://localhost:11434", client).complete(MESSAGES)
