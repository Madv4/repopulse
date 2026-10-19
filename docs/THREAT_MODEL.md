# Phase 5 threat model

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

The attack fixture `evals/fixtures/phase5_injection_cases.json` overlays 12
synthetic payloads onto the recorded issue or comment text from the pinned
`pallets/click` snapshot. The tests verify each payload is detected and that
quarantined passages never appear in LLM messages. These tests run offline with
the mock provider.

## Known limits

- The detector is heuristic. Novel wording, encoded instructions, or subtle
  task-based manipulation can evade it; benign security discussions can also
  be quarantined. It is not a semantic classifier or a proof of safety.
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
