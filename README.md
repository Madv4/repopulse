# RepoPulse

RepoPulse is a local-first project for answering questions about GitHub repositories with traceable evidence. It combines live GitHub REST data, local embeddings, a persistent vector store, a planning agent, memory, guardrails, evaluation, and per-run traces. Development is staged so each phase can be tested and committed independently.

## Status: Phase 7 complete

The project ingests GitHub issues, pull requests, comments, and commits into local ChromaDB, then answers repository questions with a LangGraph agent using eight read-only MCP tools. The agent replans when evidence is insufficient, retries tool failures with backoff, stops at configured step and token budgets, and returns a Pydantic answer with citations on every claim. Phase 4 adds in-process conversation memory and policy-gated SQLite facts and preferences. Phase 5 adds input, injection, output, citation, and action guardrails. Phase 6 adds a recorded 400-item GitHub snapshot, 55 labeled retrieval queries, 60 offline agent cases, and baseline and ablation reports. Phase 7 adds local per-run traces and an offline HTML viewer. The scripted demo remains for Phase 8.

## Quickstart

Requirements: Python 3.11+ and GNU Make. On Windows, use a Python installation with `venv` and `pip`; the Makefile uses the `py -3` launcher by default. No API key, credit card, or paid service is needed for the default local path. `make setup` installs Python packages and downloads the two free model weight sets once into `.repopulse/models`. The downloads require internet access on first setup and may take several minutes. Later inference, unit tests, and the recorded-fixture evaluation use cached weights without network access. Run `make models` to retry or refresh the model cache.

```sh
make setup
make test
make lint
make typecheck
make eval
```

For a network-free smoke check:

```sh
LLM_PROVIDER=mock .venv/bin/repopulse doctor
```

On Windows PowerShell:

```powershell
$env:LLM_PROVIDER = "mock"
.\.venv\Scripts\repopulse.exe doctor
```

`repopulse doctor --live` sends one short request to the selected provider and confirms that the configured model accepts it. The default `doctor` remains offline. Hosted providers use their free-tier quota for the live check.

The default provider is Ollama. Install Ollama separately, pull a local model such as `llama3.1:8b`, and run its local server before calling the Ollama adapter. Set `LLM_PROVIDER=groq` with `GROQ_API_KEY`, or `LLM_PROVIDER=gemini` with `GEMINI_API_KEY`, to use the optional free-tier hosted providers. The Groq default is `openai/gpt-oss-120b`, which [Groq lists on its free plan](https://console.groq.com/docs/rate-limits); its former `llama-3.3-70b-versatile` default [was retired for free-tier use](https://console.groq.com/docs/deprecations). The provider choice is explicit and tests use `mock` with no network. Copy `.env.example` to `.env` to customize settings. `.env` is ignored by Git.

Ask a question after `make setup`; local semantic search uses the cached embedding model and an ingested collection, while the other tools read public GitHub REST endpoints through the cache:

```sh
.venv/bin/repopulse ask "Which issue reported ANSI color handling?" --repo pallets/click
.venv/bin/repopulse ask "Which changes affect styling?" --repo pallets/click --max-steps 8 --token-budget 8000
```

On Windows, use `.\.venv\Scripts\repopulse.exe`. The command starts the bundled MCP server over stdio, loads the cached embedding model once before accepting tool calls, and prints progress to stderr while it plans and searches. The first `ask` in a new process can take tens of seconds while Python loads the local model; it does not download weights. Each MCP tool call has a 60-second timeout. A timed-out tool is retried within the step limit, and the answer includes a limitation if it still fails. The command prints validated, ASCII-safe JSON with `summary`, `key_changes`, `risks`, `limitations`, `degraded`, `degradation_reasons`, and `trace_id`. Each claim has a `citations` array containing retrieved issue, PR, or commit URLs. Each claim can be paraphrased but must carry a retrieved evidence ID and a verbatim quote span of at most 30 words; the verifier checks the quote in that exact passage. If an LLM step fails or its claims lack support, the answer is marked degraded and a visible `DEGRADED` line explains why. Every run has a JSONL trace under ignored `traces/`; failed model output is traced with common secrets redacted. You can run the standalone server with `python -m repopulse.mcp_server` from the virtual environment.

The default Ollama path is local and free. `LLM_PROVIDER=mock` is a network-free deterministic smoke path; its default response is not an answer and may produce an empty, degraded result. GitHub REST calls from `ask` still need connectivity unless responses are cached. Unit tests use the recorded fixture and a mock LLM, with no GitHub or model download requests. The sentence-transformers weights are fetched once by `make setup`/`make models` and then loaded from `.repopulse/models` in local-only mode; tests do not load them.

## Guardrails

All Phase 5 guards default to enabled. Set `GUARD_INPUT_VALIDATION`, `GUARD_INJECTION_DETECTION`, `GUARD_UNTRUSTED_WRAPPING`, `GUARD_OUTPUT_REDACTION`, `GUARD_STRUCTURED_VALIDATION`, `GUARD_CITATION_VERIFICATION`, or `GUARD_ACTION_GATE` to `false` in `.env` to ablate one at a time. Write requests return an `action_plan` with `dry_run: true` and require explicit confirmation; RepoPulse has no write execution tool. See [threat model](docs/THREAT_MODEL.md) for attack coverage and limitations.

## Inspect a run

Copy `trace_id` from an `ask` or `chat` answer. The CLI prints its timestamped events as JSON or creates a self-contained HTML timeline:

```sh
.venv/bin/repopulse trace show RUN_ID
.venv/bin/repopulse trace show RUN_ID --html traces/RUN_ID.html
```

On Windows, use `.\.venv\Scripts\repopulse.exe` in place of `.venv/bin/repopulse`. Open the generated HTML file in a browser; its search and event-type filter work offline. The trace records the question, plans, LLM calls and token accounting, tool arguments and attempt latencies, retrieved passages and quarantine decisions, memory reads and writes, guardrail events, and the final answer. The [recorded mock run viewer](docs/trace_viewer_example.html) and [screenshot](docs/trace_viewer_preview.png) show the layout. Regenerate that example without network access using `.venv/bin/python -m evals.render_trace_example` (Windows: `.\.venv\Scripts\python.exe -m evals.render_trace_example`). See [observability notes](docs/OBSERVABILITY.md) for the event schema and local-data limits.

![RepoPulse trace viewer preview](docs/trace_viewer_preview.png)

## Memory

The agent reads relevant active memories before planning. Add only facts or preferences you want to keep across commands:

```sh
.venv/bin/repopulse memory add "I care most about the auth rewrite." --kind preference --repo pallets/click
.venv/bin/repopulse memory list --repo pallets/click
.venv/bin/repopulse memory forget MEMORY_ID
.venv/bin/repopulse chat --repo pallets/click
```

On Windows, replace `.venv/bin/repopulse` with `.\.venv\Scripts\repopulse.exe`. `chat` keeps the last six questions and their cited URLs in memory while that process stays open; type `exit` to end the session. `ask` is a single-turn command. Long-term records live in ignored `.repopulse/memory.sqlite3` and include a timestamp, source, confidence, repository scope, and optional expiry. `memory add` always records user provenance; `--kind fact`, `--confidence 0.8`, and `--expires YYYY-MM-DD` are available. The expiry date is active through the end of that UTC day. Expired records are hidden on reads; `forget` deletes a record by ID. No GitHub tool result or issue comment is automatically saved to SQLite. See the [memory policy](docs/DESIGN.md#phase-4) for exact acceptance rules and the explicit verified-summary API.

Public GitHub reads need no token. [GitHub allows 60 unauthenticated REST requests per hour and generally 5,000 per hour with a personal access token](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api). A GitHub token is free and separate from the Groq API key. For a larger ingest window, put `GITHUB_TOKEN=...` in the local Git-ignored `.env` to raise the limit. Keep using the read-only public-repository scope you need. Cached responses are stored under `.repopulse/github-cache` by default and ignored by Git.

Ingest a live public repository, then search its local evidence:

```sh
.venv/bin/repopulse ingest pallets/click --since 2026-10-14 --until 2026-10-16 --strategy structure
.venv/bin/repopulse search "ANSI escape codes style colors" --repo pallets/click --strategy structure --rerank
```

`--since` is inclusive and `--until` is exclusive, both at midnight UTC. On Windows, use `.\.venv\Scripts\repopulse.exe` instead of `.venv/bin/repopulse`. The `ingest` command makes read-only GitHub API calls; `search` uses only local ChromaDB and cached models. Use `--strategy fixed` for the alternative collection.

Ingestion now fetches discussion and review comments through GitHub's repository-wide `since` endpoints and groups them by issue or PR, avoiding a comment request per item. It uses the PR title and body already present in GitHub's issue listing, avoiding a PR detail request per item. Paginated listings still require calls. If a run reaches GitHub's rate limit, RepoPulse prints the reset delay and exits without writing an incomplete corpus; rerun it after the reset or configure a free `GITHUB_TOKEN` in `.env`. The Groq key does not affect GitHub's request quota.

The Phase 6 evaluation uses [a pinned 400-item `pallets/click` capture](evals/fixtures/click_large_snapshot.json), [55 labeled retrieval questions](evals/datasets/retrieval_large.jsonl), and [60 agent cases](evals/datasets/agent_cases.jsonl), including 20 planted comment attacks. An additional [15-case held-out attack set](evals/fixtures/heldout_attacks.json) checks detector generalization. `make eval` regenerates [the measured Markdown report](evals/reports/latest.md) and [case-level JSON](evals/reports/latest.json). It never contacts GitHub or a hosted LLM. The two public model weight sets must already be cached by `make setup` or `make models`; offline unit tests use mock components and do not load them. Set `min_full_agent_pass_rate` in [evaluation config](evals/config.json) for the CI exit threshold. On Windows without GNU Make, run `.\.venv\Scripts\python.exe -m evals.run_suite`.

`make eval-live` is an optional ten-question Groq free-tier check with pauses between calls. Set `LLM_PROVIDER=groq` and `GROQ_API_KEY` in ignored `.env`, cache the models, and ingest `pallets/click` before running it. This makes live requests and writes `evals/reports/live_latest.md` and `live_latest.json`; it is separate from CI. On Windows without GNU Make, run `.\.venv\Scripts\python.exe -m evals.run_live`. See [evaluation methodology](docs/EVALUATION.md) for metric definitions and limitations.

## Measured Phase 6 retrieval results (mock LLM suite; ranking uses no LLM)

The offline suite measured 55 labeled queries over 400 recorded issue/PR documents. Hit rate and MRR use the first three distinct source documents.

| Chunking | Reranker | Chunks | Hit rate@3 | MRR@3 |
| --- | --- | ---: | ---: | ---: |
| Fixed | Off | 1193 | 0.800 | 0.667 |
| Fixed | On | 1193 | 0.873 | 0.806 |
| Structure-aware | Off | 1279 | 0.818 | 0.709 |
| Structure-aware | On | 1279 | 0.873 | 0.779 |

The same mock LLM run measured a 0.433 pass rate for single-retrieval naive RAG and 0.783 for the full agent across 60 cases; the planted development-attack quarantine rate was 20/20. The separate held-out check blocked 0/15. These are local recorded-fixture results, not estimates of real-model answer quality. The [evaluation notes](docs/EVALUATION.md) list all 13 full-agent failures and the held-out gap. Fixture hashes, rankings, case outcomes, and ablations are in the generated JSON. The historical Phase 2 report remains in `evals/reports/retrieval_report.md`.

## Phase roadmap

1. **Foundations:** package, config, providers, GitHub client, cache, offline tests — complete.
2. **Ingestion and retrieval:** local MiniLM embeddings, ChromaDB, two chunkers, reranking, measured retrieval evaluation — complete.
3. **MCP server and LangGraph agent loop** — complete.
4. **Short-term and SQLite memory with explicit storage policy** — complete.
5. Input, injection, output, citation, and action guardrails (complete).
6. **Deterministic answer evaluation suite and baseline comparisons** — complete.
7. **JSONL tracing and offline HTML viewer** — complete.
8. CI, polished demo, and interview documentation.

`make demo` becomes functional in Phase 8. See [design decisions](docs/DESIGN.md) for the architecture and trade-offs.

## Current architecture

```mermaid
flowchart LR
    Config[Validated settings] --> Factory[LLM provider factory]
    Factory --> Ollama[Ollama local]
    Factory --> Groq[Groq free tier]
    Factory --> Gemini[Gemini free tier]
    Factory --> Mock[Mock offline]
    Config --> GitHub[GitHub REST client]
    GitHub <--> Cache[Local JSON cache]
    GitHub --> API[GitHub REST API]
    GitHub --> Ingest[Issue PR commit ingestion]
    Ingest --> Chunk[Fixed or structure chunking]
    Chunk --> Embed[Local MiniLM embeddings]
    Embed --> Chroma[Persistent ChromaDB]
    Chroma --> Retrieve[Vector retrieval]
    Retrieve --> Rerank[Optional local cross-encoder]
    Ask[ask CLI] --> Agent[LangGraph plan and tool loop]
    Agent --> MCP[Local MCP stdio tools]
    MCP --> GitHub
    MCP --> Retrieve
    Agent --> Answer[Cited Pydantic answer]
    Memory[SQLite long-term memory] --> Agent
    Session[In-process conversation state] --> Agent
    Agent --> Trace[Local JSONL trace]
    Trace --> Viewer[Offline HTML viewer]
```

## Limitations

The agent's token budget uses a documented character-based reservation when the provider cannot report token usage, plus the provider's reported usage when available. It caps each requested completion and stops further calls at the budget, but exact tokenization varies by provider. Quote grounding permits paraphrased claims but does not prove semantic entailment. Phase 5 adds configurable prompt-injection defenses. GitHub's unauthenticated primary limit is [60 requests per hour](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api); fresh local cache entries avoid repeat calls and stale entries use conditional requests. The Chroma cache and SQLite memory can hold private local data and should be treated accordingly.
