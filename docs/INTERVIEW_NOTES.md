# RepoPulse interview notes

## The project story

I built RepoPulse to answer questions about GitHub repository activity with
inspectable source evidence. A useful answer needs more than a plausible
summary: the agent must find the right issue or PR, keep comments from
directing its behavior, cite what it actually read, and expose what happened
when a tool or model failed. I kept the live system read-only and made the
default development path free and local. The full agent uses LangGraph for
the branching plan, tool, observe, and replan loop; deterministic transport,
chunking, policy, and evaluation code stay in plain Python.

The project grew in phases. I first made GitHub reads and provider adapters
testable with injected transports. Then I added local embeddings and Chroma,
the bounded agent loop, policy-gated SQLite memory, guardrails, a pinned
offline evaluation suite, per-run traces, and finally CI and the demo. The
[README](../README.md) has the architecture and measured results; the
[evaluation notes](EVALUATION.md) explain each score and every known failure.

## Trade-offs I would explain

- **Local retrieval versus hosted search:** cached public MiniLM weights and
  Chroma make recorded evaluation runnable offline. Setup takes a download,
  and retrieval quality depends on a small, single-repository label set.
- **Bounded agent versus one-shot RAG:** planning, retries, and replans make
  behavior inspectable and resilient to some tool failures, but add latency
  and token use. The baseline comparison uses a deliberately small naive RAG,
  so it is not a claim against every well-tuned RAG design.
- **Quote grounding versus semantic verification:** a claim must cite a
  retrieved evidence ID and a short span that occurs in that passage. This
  rejects invented quotes while allowing paraphrase. It cannot prove that
  the paraphrase is entailed by the quote.
- **Memory with an explicit write gate:** user-stated facts and explicit
  verified summaries may enter SQLite. GitHub content never gets promoted
  automatically, so a poisoned comment cannot persist as a user memory.
- **Heuristic plus optional LLM screening:** the heuristic is cheap and
  offline. It blocked the development payloads but missed held-out variants
  and once rejected a benign Click comment with a `#keywords` link. The
  optional live classifier catches more of the frozen attack sets, at the
  cost of provider calls, latency, and a separate failure mode. The benign
  audit used a different free Groq model from the attack screen, so I would
  not present their rates as a same-model precision-recall curve.
- **Read-only action gate:** the agent can plan a write, but cannot execute
  one. That narrows the tool-hijack surface and also means the live attack
  test cannot measure unauthorized writes that the architecture forbids.

## Hardest problems and what changed

1. A live Click ingest hit GitHub's unauthenticated rate limit while reading
   comments per issue. I switched to repository-wide comment endpoints,
   grouped by parent, and kept conditional local caching. Large windows can
   still hit pagination or rate limits.
2. Live model output sometimes failed structured planning, and plausible
   cited claims failed exact quote checks. I added structured JSON mode where
   supported, validation feedback, one synthesis retry, Unicode/whitespace
   normalization, and trace records for rejected claims. Fabricated quotes
   still fail the verifier.
3. The first retrieval fixture was too easy. A larger pinned Click capture,
   paraphrased labels, and near-duplicate distractors made variant differences
   visible. It remains one public repository, not a broad benchmark.
4. The development injection set made the heuristic look stronger than it
   was. Frozen held-out sets exposed misses. I kept those misses in the
   reports and added an optional classifier without tuning it on the held-out
   v2 set.
5. A requested benign audit ran into the free Groq model's rolling token
   quota. I used another available free Groq model for the entire fixed
   sample, recorded that model in the report, and called out the resulting
   comparison limit. The sample produced fewer false positives than the ten
   examples requested; I listed the one observed example instead of inventing
   the rest.

## What the results do and do not say

The offline agent numbers come from a scripted mock LLM and rule-based labels.
They test retrieval, citations, retry paths, and guardrail control flow; they
do not estimate general live-model answer accuracy. The attack-screening
numbers count quarantined planted comments, while the live attack run scores
observable answer leakage, unauthorized tool calls, and attacker text copied
as an instruction-following answer. The unguarded live run also resisted the
observed attacks, so I would not claim an observed live safety improvement
from these cases. The scorer can miss subtle steering. Every metric in the
README is regenerated from saved report JSON rather than hand-entered.

## What I would do next

I would gather multiple repositories and independently labeled answer claims,
review a larger same-model benign set alongside new attacks, test indirect
instructions across several retrieval turns, and assess semantic claim
support with human review. I would also benchmark cost and latency under a
real provider after normalizing call counts, add trace retention controls,
and replace the local pending CI badge with GitHub's native status badge once
the repository has a remote and a workflow run.
