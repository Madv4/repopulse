# Evaluation methodology

## Phase 2 retrieval benchmark

The committed fixture `evals/fixtures/click_snapshot.json` records selected fields from real public `pallets/click` REST responses, with a capture timestamp and the exact API source. It contains curated issue and PR summaries, PR details, comments, and commit summaries. `evals/capture_snapshot.py` documents the capture procedure; rerunning it later creates a new snapshot and requires fresh label review. The committed fixture is the pinned corpus for the benchmark. No GitHub calls occur in `make eval`.

The labeled dataset is `evals/datasets/retrieval.jsonl`. Each case has a question and one or more source URLs judged relevant from the captured records. Labels were chosen before the final comparison. Queries include title-level facts, commit-level facts, and details found inside comments. The fixture-integrity test checks that every labeled URL exists in the snapshot.

The runner evaluates four configurations: fixed chunks and structure-aware chunks, each with and without local cross-encoder reranking. It rebuilds the fixture repository in a dedicated local Chroma directory on each run. Both strategies use the same source documents, embedding model, ChromaDB vector search, candidate pool, `k=3`, and source-level deduplication. The reranker scores at most 20 retrieved passages per query. `hit rate@3` is the share of queries with any relevant source among the first three distinct sources. `MRR@3` averages the reciprocal rank of the first relevant source, assigning zero to a miss. The report includes actual document and chunk counts and the SHA-256 of the fixture and labels.

Run `make setup` once to cache the public model weights. Thereafter `make eval` uses the cache in local-only mode and writes `evals/reports/retrieval_report.md` plus `evals/reports/retrieval_results.json`. Unit tests inject fake vectors and never load model weights. The reports are generated from returned rankings; numbers are not hardcoded.

This is a small curated smoke benchmark on one repository. It can show regressions in this pipeline, but does not estimate general retrieval quality, answer accuracy, or prompt-injection resistance. The full answer and guardrail evaluation belongs to Phase 6.

The Phase 2 retrieval fixture is too easy: its near-perfect scores do not discriminate between the chunking and reranking variants. Phase 6 will add a larger, harder retrieval set built from a recorded snapshot of a real public repository, with paraphrased queries and near-duplicate distractors. The new set will remain offline and use reviewed source-URL labels.
