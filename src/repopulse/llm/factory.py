"""Select an LLM provider from validated settings."""

import httpx

from repopulse.config import Settings
from repopulse.llm.base import LLMProvider
from repopulse.llm.gemini import GeminiProvider
from repopulse.llm.groq import GroqProvider
from repopulse.llm.mock import MockProvider
from repopulse.llm.ollama import OllamaProvider


def create_provider(settings: Settings, *, client: httpx.Client | None = None) -> LLMProvider:
    """Construct the requested provider; only the selected provider needs a key."""
    if settings.llm_provider == "mock":
        return MockProvider(fallback=settings.mock_response)
    if settings.llm_provider == "ollama":
        return OllamaProvider(settings.ollama_model, settings.ollama_base_url, client)
    if settings.llm_provider == "groq":
        key = settings.groq_api_key.get_secret_value() if settings.groq_api_key else ""
        return GroqProvider(settings.groq_model, key, client)
    key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else ""
    return GeminiProvider(settings.gemini_model, key, client)
