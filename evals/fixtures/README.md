# Recorded GitHub fixture

`click_snapshot.json` is a committed, timestamped capture of selected public `pallets/click` GitHub REST response fields. It includes issues, PR details, issue comments, PR review comments, and commits. The selection was curated for a small retrieval benchmark; it is not a complete repository archive. API fields and comment text were preserved, while unrelated response fields were omitted to keep the fixture reviewable.

`python -m evals.capture_snapshot` is an optional, **networked** recapture command. It uses the public GitHub REST API and can consume unauthenticated rate-limit allowance. It must not run in unit tests or CI. A new capture changes the corpus and requires review of every relevance label before reporting new metrics.
