# Retrieval evaluation

Measured by `make eval` on recorded GitHub REST fields; no GitHub or LLM network calls.
Fixture: `pallets/click`, captured 2026-10-19T06:26:22.403901+00:00; SHA-256 `3d39ff0c6a8ada85bdd0c0145b7ab8f0cf99f3c73996f15046617918156113d7`.
Dataset: 21 labeled queries; SHA-256 `4048d186ef6c76ac6c2bd4e0c546906513a7bb88aaec3a468483ef225be1c5ac`.
Embedding: `sentence-transformers/all-MiniLM-L6-v2` at `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
Reranker: `cross-encoder/ms-marco-MiniLM-L6-v2` at `233902d25c440f23af6f7d6e94d2946bac0bee0a`.
Packages: ChromaDB 1.5.9, sentence-transformers 6.1.0.

| Chunking | Reranker | Documents | Chunks | Hit rate@3 | MRR@3 |
| --- | --- | ---: | ---: | ---: | ---: |
| fixed | off | 20 | 85 | 1.000 | 1.000 |
| fixed | on | 20 | 85 | 1.000 | 1.000 |
| structure | off | 20 | 96 | 1.000 | 0.976 |
| structure | on | 20 | 96 | 1.000 | 1.000 |

Hit rate@3 is the fraction of queries with a labeled source among the first three distinct source documents. MRR@3 is mean reciprocal rank of the first labeled source in those results. Labels were written before running the comparison. Both chunkers use 500-character limits; fixed chunking overlaps 80 characters. The cross-encoder reranks up to 20 vector candidates. This small curated dataset is a retrieval smoke benchmark, not a general accuracy claim.

Per-query ranked URLs and metrics are in `retrieval_results.json`.
