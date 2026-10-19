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

The Phase 1 commit installed no embedding, vector-store, LangGraph, MCP, tracing, or CI dependencies. They enter in their corresponding phases. Phase 2 adds local MiniLM and ChromaDB; SQLite memory remains for Phase 4.

### Sources checked

- [GitHub REST rate limits](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api)
- [GitHub REST pagination](https://docs.github.com/en/rest/using-the-rest-api/using-pagination-in-the-rest-api)
- [GitHub conditional request guidance](https://docs.github.com/en/rest/using-the-rest-api/best-practices-for-using-the-rest-api)
- [Groq chat completions API](https://console.groq.com/docs/api-reference)
- [Gemini generateContent API](https://ai.google.dev/gemini-api/docs/generate-content)

## Phase 2

### Source normalization and date windows

The initial Phase 2 `RepositoryIngestor` read GitHub's issue listing (which includes PR summaries), PR details, issue discussion comments, PR line review comments, and commits. It selects records with creation, update, close, or merge timestamps inside a half-open UTC window and includes comments created in that window. The live endpoint's `since` filter selects records updated after the start date. This avoids retrieving the entire repository for common recent windows, but a long or very active window can still hit the client's explicit pagination cap. GitHub's current issue body is not a historical reconstruction of its state at the window end; a future timeline-based snapshot would be needed for that. The fixture evaluation avoids this drift by using a committed capture. The rate-limit repair below changes how PR and comment records are fetched while retaining their normalized fields.

All sources become a `SourceDocument` with canonical ID, URL, title, body, author, dates, labels and comments. Each chunk keeps its parent issue, PR or commit URL and metadata. Comment text stays distinguishable by author and comment URL in the passage, even though the parent URL is the primary citation key.

### Chunking and local search

`FixedChunker` uses 500-character windows with 80-character overlap. `StructureChunker` packs Markdown paragraphs, headings, and whole comment blocks up to 500 characters, only slicing oversized blocks. Both implement the same `Chunker` interface. ChromaDB has a separate persistent collection per strategy. Metadata includes source type, number, URL, author, created and updated dates, labels, source ID, repo, strategy and chunk index. Chroma metadata does not store arrays directly, so labels are JSON-encoded. Reingestion replaces chunks belonging to one source.

The embedder is explicitly local `sentence-transformers/all-MiniLM-L6-v2`, with vectors supplied to ChromaDB. Chroma's default embedding function is never invoked. The optional cross-encoder is `cross-encoder/ms-marco-MiniLM-L6-v2`; it scores query/passage pairs after vector retrieval. Retrieval deduplicates by source document so long issues do not occupy every returned rank. The current candidate pool has a fixed bound; for very long corpora it can return fewer than `k` distinct sources, a known limitation.

`make setup` downloads both public model weight sets once into `.repopulse/models`. The configured Hugging Face revisions are pinned to the actual downloaded snapshots, and the evaluation report records both hashes. Runtime inference and `make eval` request cached files only and fail with a clear `make models` instruction when weights are absent. Unit tests inject a deterministic tiny embedder and reranker, so they never download a model or contact GitHub. The recorded retrieval fixture has selected real `pallets/click` REST fields; no fixture text is sent to a hosted service.

The NumPy dependency is capped below 2.5 because the 2.5.4 stubs use syntax mypy rejected under the project's Python 3.11 type-check target. Mypy checks all project source and evaluation code strictly while skipping deep traversal into the large Torch, Transformers, NumPy and Chroma dependency trees.

Plain Python remains the fit for ingestion, chunking, and retrieval because these are deterministic data transformations. LangGraph is reserved for the branching agent loop in Phase 3.

### Sources checked

- [Chroma persistent client](https://docs.trychroma.com/reference/python)
- [SentenceTransformer model loading and local-only mode](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html)
- [CrossEncoder model loading and local-only mode](https://www.sbert.net/docs/package_reference/cross_encoder/model.html)
- [GitHub issue REST API](https://docs.github.com/en/rest/issues/issues)
- [GitHub issue comments](https://docs.github.com/en/rest/issues/comments)
- [GitHub PR review comments](https://docs.github.com/en/rest/pulls/comments)

## Phase 3

### MCP boundary and agent graph

The official `mcp` Python SDK/FastMCP server exposes eight read-only tools over stdio: `search_issues`, `get_issue`, `get_pull_request`, `get_pr_files`, `list_commits`, `list_milestones`, `list_releases`, and `semantic_search`. Handlers return a small typed evidence envelope with bounded excerpts and source URLs. The CLI starts the same server as a subprocess and calls it through an MCP `ClientSession`; the agent sees only a `ToolGateway` protocol. Unit tests replace that gateway with recorded results, and an integration test starts a fixture-backed server over real MCP stdio. GitHub list tools request one bounded page. The server preloads cached local embeddings before accepting calls; semantic search opens Chroma when invoked.

LangGraph owns the explicit `plan -> select -> execute -> observe -> decide -> synthesize` transitions. The decision can continue, replan, synthesize after enough evidence, or stop. At most three replans follow the initial plan. Every MCP attempt, including retry attempts, consumes a step. Tool errors use two exponential-backoff retries, then add a limitation and allow the graph to use other evidence. A selected tool name is validated against the eight-tool allowlist; the requested repository is replaced with the validated CLI repository before execution.

The agent reserves an estimated prompt budget from message length, sends the remaining per-call output cap to the selected provider, and counts the greater of estimated or provider-reported usage. It will not make another LLM call when that accounting budget cannot fit the prompt. This is a hard control-flow budget, but provider tokenizers differ, so a single call's reported usage can exceed the estimate; the agent then stops. The mock provider can return explicit usage in tests. The maximum tool-step cap is exact.

The synthesizer validates an `Answer` Pydantic object containing a summary claim, key-change claims, risk claims, and limitations. Each retained claim must have at least one citation URL retrieved from an issue, pull request, or commit. Milestone and release records can guide search but cannot serve as final citations. The initial Phase 3 version used a retrieved title when synthesis failed and checked URL membership only; the follow-up below replaces that behavior with extractive support checking and an explicit degraded answer.

All Phase 3 tests are offline and use the pinned recorded `pallets/click` fields plus the scripted mock LLM. The fixture intentionally lacks PR-file, milestone, and release lists, so tests verify empty outputs for those tools instead of inventing records. `make eval` remains the Phase 2 retrieval evaluation. Its near-perfect small-fixture scores are not treated as comparative evidence; the planned harder Phase 6 set is described in `docs/EVALUATION.md`.

### Sources checked

- [MCP Python SDK server and client examples](https://github.com/modelcontextprotocol/python-sdk)
- [LangGraph StateGraph reference](https://langchain-ai.github.io/langgraph/reference/graphs/)
- [Groq chat completions API](https://console.groq.com/docs/api-reference)
- [Gemini generateContent API](https://ai.google.dev/api/generate-content)

## Rate-limit repair after Phase 3

A live `pallets/click` ingest exhausted GitHub's unauthenticated hourly quota while fetching per-issue comments. The attempted run had populated many independent comment-list cache entries; the 5-minute cache TTL could not make the next run reliably resume after an hourly reset. Ingestion now reads repository-wide issue discussion comments and PR review comments with `since`, groups their canonical parent URLs by number, then attaches only comments created within the requested window. It uses PR title, body, author, and dates from the issue listing, which already carries those fields, so it does not request a PR detail per item. The separate MCP `get_pull_request` tool still fetches details when the agent asks for them. This removes repeated comment and PR-detail requests without dropping the normalized fields. The recorded fixture adapter derives parent URLs from its recorded keys, preserving its existing contents and offline evaluation behavior.

The CLI now prints a rate-limit recovery instruction instead of a traceback. The optional `GITHUB_TOKEN` is a GitHub credential independent of the LLM provider key; no paid service is required. A run that still exceeds the quota fails before any Chroma writes, so it cannot report a partial ingest as complete.

Sources: [repository issue comments](https://docs.github.com/en/rest/issues/comments), [repository PR review comments](https://docs.github.com/en/rest/pulls/comments), [GitHub REST rate limits](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api).

## Follow-up after Phase 3

The earlier issue-identification filter counted words shared with the question. That could accept an unrelated source, and it applied only to one question shape. The replacement is general and deliberately conservative: after Unicode and whitespace normalization, a retained claim's full text must occur in the title or excerpt of **each** retained citation. A URL alone never establishes support. The LLM is asked to copy source wording; unsupported paraphrases are dropped. If none survive, no answer claim is issued and the result is marked degraded. This extracts only explicit evidence and does not attempt semantic entailment; broader paraphrase verification remains for Phase 5.

An LLM planning, selection, decision, or synthesis fallback sets `degraded` and records safe reasons in the answer. The CLI prints a `DEGRADED` line and writes those reasons, without prompts or source text, to a local JSONL trace under ignored `traces/`. Full run tracing remains a Phase 7 task. `doctor --live` makes a single completion request to the selected model; success checks that the provider accepts the configured model name. The ordinary `doctor` makes no request.

## Phase 4

### Memory policy

Long-term memory is an explicit, local SQLite store. A record has an opaque ID, text, kind (`preference`, `fact`, or `summary`), source (`user` or `agent_verified`), optional repository scope, confidence in `[0, 1]`, a UTC creation timestamp, optional expiry, and source URLs for verified summaries. Confidence records the strength assigned by the trusted writer; it is not an evaluated probability. Active reads hide expired records, and `memory forget` physically deletes a record.

The write rules are:

1. `memory add` is an explicit user action. It may save a user-stated fact or preference of 12–500 characters. The user can scope it to a validated `owner/name` repository and set an expiry. Trivial empty or short text, invalid scope, out-of-range confidence, and past expiry are rejected.
2. An agent-verified summary may be saved only through the explicit `MemoryStore.add_verified_summary` API. Every cited URL must be in supplied evidence, and the full summary text must occur in each cited title or excerpt under the same conservative extractive rule used for final claims. This is an explicit promotion operation; the agent's normal `ask` and `chat` paths never call it.
3. GitHub issues, pull requests, commits, and comments are untrusted tool output. The policy rejects `source=github` outright. No retrieved content is automatically written to SQLite, even when an answer quotes it. Transient observations, raw tool results, unsupported paraphrases, and instructions planted in issue comments are ignored for long-term storage.

The agent reads up to five active memories before each plan. Repository-scoped records and preferences receive a small deterministic priority; word overlap with the question provides additional ranking. Global facts with no overlap are not injected. Memories guide planning only and are not answer citations: final claims still require retrieved issue, PR, or commit evidence. This simple local ranking has no network or embedding dependency. The SQLite file defaults to `.repopulse/memory.sqlite3`, outside Git, and can contain private user statements.

Short-term memory is a six-turn in-process window. It keeps user questions and cited URLs, not raw GitHub passages or comments. `repopulse chat --repo owner/name` reuses it across questions in one process; `ask` starts a new one-turn session. The window is discarded on exit. Plain Python and SQLite are a better fit for this deterministic policy and persistence than additional LangGraph nodes; LangGraph continues to handle the branching search loop.
