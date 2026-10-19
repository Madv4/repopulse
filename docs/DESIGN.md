# Design decisions

## Scope and phases

RepoPulse follows the requested eight-phase delivery order. This document records implementation choices as they are made. Phase 1 intentionally stops at package infrastructure, provider adapters, and GitHub data transport. Later architecture is a plan, not a claim of implemented behavior.

## Phase 1

### Provider contract

`LLMProvider.complete()` accepts ordered Pydantic `ChatMessage` objects and returns a Pydantic `LLMResponse`. All four providers expose the same method. Ollama uses its local `/api/chat` API. Groq uses its OpenAI-compatible chat completions endpoint. Gemini uses `generateContent` with separate system instructions and content turns. The mock supports an ordered script, a responder function, or a fixed fallback; it has no network path and records calls for assertions. Provider HTTP errors return safe status-only messages and never include response bodies or credentials.

Provider choice is validated by `Settings` via `LLM_PROVIDER`; only the selected hosted adapter requires a key. Token counts are reported only when supplied by the provider. The mock returns zero reported tokens rather than pretending its whitespace count is real usage.

### GitHub transport and cache

The client is synchronous and read-only. It uses `httpx` so tests can inject `MockTransport` without network. A `GET` response is cached by URL and a one-way digest of the authentication scope. Credentials are not written to the cache. Fresh entries are served locally; stale entries send `If-None-Match` and `If-Modified-Since` when available. A `304` refreshes the cache timestamp. Cache writes use atomic replacement. The cache can contain private repository data if a PAT grants access, so its directory must be treated as private local data.

The default cache TTL is five minutes. This balances live data against GitHub's unauthenticated limit; users can set `GITHUB_CACHE_TTL_SECONDS=0` for revalidation on every call. According to [GitHub's guidance](https://docs.github.com/en/rest/using-the-rest-api/best-practices-for-using-the-rest-api), an authenticated conditional request returning `304` does not count against the primary limit. We do not assume that benefit for unauthenticated requests.

The client follows the API's [`Link` pagination header](https://docs.github.com/en/rest/using-the-rest-api/using-pagination-in-the-rest-api), limits pages, detects loops, and rejects links that leave the configured API origin. A `403`/`429` rate-limit response raises `GitHubRateLimitError` with a retry delay derived from `Retry-After` or `X-RateLimit-Reset`; it does not hammer the endpoint. Transient transport errors and `5xx` responses get at most two retries with bounded exponential backoff. Error messages avoid URLs, headers, tokens, and response bodies.

### Dependencies and reproducibility

The Phase 1 runtime depends only on `httpx`, `pydantic`, and `pydantic-settings`. The package uses the `src/` layout and supports Python 3.11+. `pytest`, `ruff`, and `mypy` are dev dependencies. Unit tests use only deterministic HTTP fixtures. The Makefile works with a Python virtual environment on Unix and Windows GNU Make. It reserves `eval` and `demo` target names without suggesting those later phases are complete.

Plain Python is the clearer fit for configuration, provider adapters, and HTTP transport: these operations are single calls with no branching agent state to orchestrate. LangGraph enters in Phase 3 for the plan, tool, observe, and replan loop, where explicit state transitions matter.

No embedding, vector-store, LangGraph, MCP, tracing, or CI dependencies are installed yet. They enter in the corresponding phases to keep each step reviewable and avoid imposing large downloads during foundation setup. The future embedding model is local `all-MiniLM-L6-v2`, with persistent ChromaDB and SQLite as specified in the project brief.

### Sources checked

- [GitHub REST rate limits](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api)
- [GitHub REST pagination](https://docs.github.com/en/rest/using-the-rest-api/using-pagination-in-the-rest-api)
- [GitHub conditional request guidance](https://docs.github.com/en/rest/using-the-rest-api/best-practices-for-using-the-rest-api)
- [Groq chat completions API](https://console.groq.com/docs/api-reference)
- [Gemini generateContent API](https://ai.google.dev/gemini-api/docs/generate-content)
