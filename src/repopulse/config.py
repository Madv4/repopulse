"""Validated, environment-driven application configuration."""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings; optional secrets are never included in repr output."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: Literal["ollama", "groq", "gemini", "mock"] = "ollama"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.1:8b"
    groq_api_key: SecretStr | None = Field(default=None, repr=False)
    groq_model: str = "openai/gpt-oss-120b"
    gemini_api_key: SecretStr | None = Field(default=None, repr=False)
    gemini_model: str = "gemini-2.5-flash"
    mock_response: str = "Mock response"
    github_token: SecretStr | None = Field(default=None, repr=False)
    github_cache_dir: Path = Path(".repopulse/github-cache")
    github_cache_ttl_seconds: int = Field(default=300, ge=0)
    github_timeout_seconds: float = Field(default=15, gt=0)
    github_max_retries: int = Field(default=2, ge=0, le=5)
    model_cache_dir: Path = Path(".repopulse/models")
    chroma_dir: Path = Path(".repopulse/chroma")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_revision: str = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
