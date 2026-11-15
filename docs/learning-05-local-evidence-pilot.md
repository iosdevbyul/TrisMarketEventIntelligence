# Learning-05 — controlled real local Gemma evidence pilot

Uses the existing immutable 482-article cohort and the already exported
48-candidate human-review sample, **without asking anyone to label news**.
This is an independent research pilot, not an alternative route for writing
a human review or changing training splits.

The existing `materiality-local` command correctly requires its entire
research cohort and cannot run its 20-article maximum against 482 articles.
This separate, bounded research path validates the *original full cohort*
and *rebuilds/verifies the deterministic candidate sample*, then sends only
the first 1–3 sampled RSS headline+summary records to a **previously
installed** local Ollama model (default `gemma4:e2b`).

Before any model call, it checks complete source text SHA/capture hash
metadata and sample provenance. It then accepts only model outputs with
the original article ID and source text SHA, a verbatim source
evidence excerpt and a source-present entity. The existing Pydantic
NEWS hypothesis schema and unassessed/uncertain consistency apply.

- Contract accepted and a materiality hypothesis offered:
  `ACCEPTED_UNVERIFIED_HYPOTHESIS`, **not** a proven correct inference.
- Contract accepted but magnitude withheld: `ACCEPTED_ABSTAINED`.
- Invalid response, invented entity/excerpt, or request failure:
  `REJECTED_OR_MODEL_RUNTIME_FAILED` with a fixed `failureCode` such as
  `MODEL_TIMEOUT`, `MODEL_SCHEMA_INVALID`, `SOURCE_EVIDENCE_MISMATCH`,
  or `ABSTENTION_DIRECTION_MISMATCH`. Counts include the failure reason
  separately. No raw model output or exception strings appear in stdout.
  No invented fallback gold.
- Missing local model: `LOCAL_MODEL_NOT_INSTALLED`, no pull/download.

A local 0600 create-only JSON contains source excerpts and unverified
hypotheses; it must remain private. Terminal stdout contains only
counts and status, never source text or the inferred model rationale.
No bulk full-cohort training, no independent semantic accuracy score,
no market prices/returns/volume/OHLCV, no auto-promotion, no persistent
worker, no reserved TEST access, no production DB writes.

## Preview (zero model calls)

```sh
bash scripts/news-learning-review.sh --update evidence-pilot --preview \
  --cohort "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-cohort-20261115.json" \
  --sample "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-review-sample-20261115.json" \
  --model gemma4:e2b --limit 2
```

## Opt-in real Gemma 4 pilot

```sh
bash scripts/news-learning-review.sh evidence-pilot --run \
  --cohort "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-cohort-20261115.json" \
  --sample "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-review-sample-20261115.json" \
  --model gemma4:e2b --limit 2 \
  --output "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-evidence-pilot-20261115.json"
```

Only execute when the operator explicitly decides to run local inference.
The model must already be installed and Ollama running at 127.0.0.1:11434.
If the original local output already exists, use a new filename rather than
overwriting it. Inspect outcomes for operational quality, not model accuracy.

## After a rejection on the earlier pilot

Old JSON files cannot reconstruct the underlying exception; two rejected
calls in the original run are still **unknown cause**. Run the same 2
candidates again with a new output filename after pulling the fix:

```sh
bash scripts/news-learning-review.sh --update evidence-pilot --run \\
  --cohort "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-cohort-20261115.json" \\
  --sample "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-review-sample-20261115.json" \\
  --model gemma4:e2b --limit 2 \\
  --output "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-evidence-pilot-diagnostic-20261115.json"
```

A failed model output is *never* retried automatically and none of the
failure codes establish factual accuracy or model intelligence. Preserve
the prior attempt's original private file unchanged.
