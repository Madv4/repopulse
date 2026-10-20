# Prompt-injection threat model

RepoPulse reads public GitHub issues, PRs, comments, commits, and local index
passages. Anyone able to edit those sources can plant text the agent later
retrieves. User questions and provider responses can also contain sensitive
data. The protected assets are the user's secrets and contact details, the
integrity of the answer, the configured repository scope, and local state.

## Boundaries and defenses

| Attack | Boundary |
| --- | --- |
| Oversized or out-of-scope input | Query length, public `owner/name` format, and scope wording checks; tool calls are pinned to the chosen repo. |
| Instruction override, fake system/developer messages, tool hijack | Pattern and density detector quarantines suspicious evidence before any LLM prompt. Remaining evidence is serialized in delimited `untrusted_data` blocks; the system prompt says to treat it only as data. |
| Indirect, multilingual, or split instructions | Optional provider-neutral LLM classifier reviews heuristic-clean passages and combined discussion passages from the same source. It is off by default, including in CI and mock runs. Validated JSON decisions are cached by passage hash and provider/model. |
| Secret exfiltration through GitHub text or generated answers | The agent's tool allowlist is read-only; common email, phone, and token formats are redacted from final answer fields and validation traces. |
| Fabricated citations and invented quotes | Each claim names retrieved passage IDs and quotes at most 30 words verbatim; the verifier resolves URLs from those IDs and drops claims with missing or invented quotes. |
| Malformed model output | Provider JSON mode, Pydantic parsing, one error-guided retry, then an explicit degraded fallback. Failed raw output is recorded locally with secrets redacted. |
| Write request disguised as a question | Action gate returns an inert dry-run plan requiring explicit confirmation. No write-capable MCP tool or execution path is registered. |

Each boundary has its own `GUARD_*` boolean setting in `.env`: `GUARD_INPUT_VALIDATION`,
`GUARD_INJECTION_DETECTION`, `GUARD_UNTRUSTED_WRAPPING`,
`GUARD_OUTPUT_REDACTION`, `GUARD_STRUCTURED_VALIDATION`,
`GUARD_CITATION_VERIFICATION`, and `GUARD_ACTION_GATE`. All default to `true`.
Set one to `false` for a Phase 6 ablation. Disabling structured validation
removes the corrective retry; minimum Pydantic shape checks remain necessary
for graph execution. Disabling the action gate cannot enable writes because
the tool surface remains read-only.

The optional `GUARD_INJECTION_CLASSIFIER=true` setting adds a second screen after
the pattern detector. The classifier receives the source title and excerpt as
lower-trust data, returns a Pydantic-validated JSON decision, and never gets
actual local credentials. A chunk's decision is cached under a hash of its
content plus the provider, model, and prompt version; raw source text is not
stored in this cache. Multiple passages from the same issue are also reviewed
together to catch split instructions. `INJECTION_CLASSIFIER_CACHE_DIR` changes
the ignored local cache location. The agent traces its decision, latency, cache
hit, and provider token counts. The classifier consumes the agent's budget;
when it cannot run, the trace and degraded answer say that only heuristic
filtering was used. It is intentionally disabled by default for offline mock
tests and CI, and constructing it with the mock provider is rejected. It was
designed from attack classes rather than the 15 held-out v1 examples; no prompt
or threshold was tuned against those examples. When the selected provider is
hosted, the title and excerpt being reviewed are sent to that provider. Keep
this option off for material you do not intend to share with the provider.

The attack fixture `evals/fixtures/phase5_injection_cases.json` overlays 12
synthetic payloads onto the recorded issue or comment text from the pinned
`pallets/click` snapshot. The tests verify each payload is detected and that
quarantined passages never appear in LLM messages. These tests run offline with
the mock provider.

## Known limits

<!-- measured-attack-limits:start -->
The generated live Groq behavior report measured the following attack successes with default guardrails ON versus OFF:

- dev: ON 0/20 (live Groq); OFF 0/20 (live Groq).
- heldout_v1: ON 0/15 (live Groq); OFF 0/15 (live Groq).

The optional classifier plus heuristic left these attack comments unscreened:

- dev: none (live Groq classifier).
- heldout_v1: `heldout-12` (split_comments) (live Groq classifier).
- heldout_v2: `heldout-v2-02` (editorial_pressure), `heldout-v2-09` (markdown_table) (live Groq classifier).

These are observed results on frozen synthetic fixtures, not proof of general safety. The live behavior scorer can miss paraphrased compliance.
<!-- measured-attack-limits:end -->

- The detector is heuristic. Novel wording, encoded instructions, or subtle
  task-based manipulation can evade it; benign security discussions can also
  be quarantined. It is not a semantic classifier or a proof of safety.
- The optional LLM classifier can also be manipulated by the text it reviews,
  make false positive or false negative decisions, exhaust a provider's free
  quota, or miss relationships beyond the bounded combined excerpt. A cache
  preserves its first decision for a given content hash and model until the
  prompt version changes or the cache is cleared. A provider or budget failure
  degrades to heuristic-only filtering; it does not silently assert safety.
- Delimiters and prompt instructions reduce authority confusion but do not
  guarantee that a model will ignore all malicious source text.
- A valid quote proves the phrase occurred in the retrieved passage, not that
  a free-form claim is logically entailed. The quote can be selective or
  misleading. Human review is needed for high-stakes conclusions.
- Regex redaction misses uncommon secret and phone formats and can remove
  harmless text. It is not a data-loss-prevention system. Raw GitHub content
  and the SQLite memory store can still contain sensitive data on disk.
- The action gate recognizes direct English write requests. An indirect or
  ambiguous request may pass to the read-only agent, but cannot invoke a
  write tool. A future write implementation must enforce confirmation at its
  own execution boundary.
- JSON mode availability varies by provider and model. Pydantic validation
  and the degraded fallback remain the final application boundary.
