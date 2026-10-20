"""README headline numbers are composed from report fields, not constants."""

from evals.publish_readme_results import render


def test_results_tables_use_saved_values_and_label_providers() -> None:
    offline = {
        "metric_source": "mock LLM",
        "retrieval": [{"strategy": "fixed", "reranker": False, "chunks": 7,
                       "hit_rate_at_k": 0.25, "mrr_at_k": 0.125}],
        "naive_rag": {"cases": [{"pass": False}], "case_count": 1,
                      "pass_rate": 0.0, "citation_correctness_rate": 0.5,
                      "injection_block_rate": 0.0},
        "full_agent": {"cases": [{"pass": True}], "case_count": 1,
                       "pass_rate": 1.0, "citation_correctness_rate": 1.0,
                       "injection_block_rate": 1.0},
    }
    attacks = {"metric_source": "live Groq", "summaries": [
        {"dataset": "dev", "guardrails": "off", "attack_successes": 1,
         "case_count": 2, "attack_success_rate": 0.5},
    ]}
    screening = {"summary": [
        {"dataset": "dev", "heuristic_blocked": 1, "extra_classifier_blocks": 1,
         "combined_blocked": 2, "case_count": 3, "combined_block_rate": 2 / 3},
    ]}
    benign = {"metric_source": "heuristic only; live Groq classifier",
              "classifier_model": "fixture-model", "counts": {
                  "sample": 5, "heuristic_only_mode": 1,
                  "classifier_only_mode": 2, "both_screens_mode": 3,
              }}
    text = render(offline, attacks, screening, benign)
    assert "mock LLM" in text
    assert "live Groq" in text
    assert "| Fixed | Off | 7 | 0.250 | 0.125 |" in text
    assert "| Full agent | 1/1 | 1.000 | 1.000 | 1.000 |" in text
    assert "| dev | off | 1/2 | 0.500 |" in text
    assert "| Heuristic + live Groq classifier | 3/5 | 0.600 |" in text
    assert "unguarded live Groq run also resisted" not in text
