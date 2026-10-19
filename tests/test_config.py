"""Configuration and provider selection tests."""

import pytest
from pydantic import ValidationError

from repopulse.config import Settings
from repopulse.llm import ProviderError, create_provider


def test_defaults_are_local_and_free() -> None:
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "ollama"
    assert settings.github_token is None
    assert settings.github_cache_ttl_seconds == 300


def test_provider_selection_and_missing_keys() -> None:
    assert create_provider(Settings(_env_file=None, llm_provider="mock")).name == "mock"
    assert create_provider(Settings(_env_file=None, llm_provider="ollama")).name == "ollama"
    with pytest.raises(ProviderError, match="GROQ_API_KEY"):
        create_provider(Settings(_env_file=None, llm_provider="groq"))
    with pytest.raises(ProviderError, match="GEMINI_API_KEY"):
        create_provider(Settings(_env_file=None, llm_provider="gemini"))
    assert create_provider(
        Settings(_env_file=None, llm_provider="groq", groq_api_key="fake")
    ).name == "groq"
    assert create_provider(
        Settings(_env_file=None, llm_provider="gemini", gemini_api_key="fake")
    ).name == "gemini"


def test_invalid_settings_and_secrets() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_provider="unknown")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, github_cache_ttl_seconds=-1)
    settings = Settings(_env_file=None, github_token="test-secret")
    assert "test-secret" not in repr(settings)
