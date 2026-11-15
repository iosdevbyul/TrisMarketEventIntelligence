# Learning-05: reproducible private news importance model candidates

This step persists **research-only** model weights from an already recorded
Learning-05 CPU experiment. It does not produce a production model, certify
improved accuracy or activate any Mac daemon.

The existing `news_research_worker.py` and exported candidate share the same
`NewsImportanceModel` tokenizer inputs and multinomial Naive Bayes prediction
implementation. TRAIN is confined to human NEWS importance judgments available
before the training cutoff; VALIDATION is for descriptive Log Loss; reserved
TEST gold is never used for training, model selection or replay.

## Export and reproduce

The existing `scripts/news-learning-review.sh` now supports `candidate`.
Examples below are **only for later**, when genuine private news review and
a completed research experiment are available:

```sh
bash scripts/news-learning-review.sh --update candidate --export \
  --cohort /private/news-cohort.json \
  --holdout /private/news-holdout-with-human-review.json \
  --review /private/news-human-review.json \
  --journal "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-research.sqlite3" \
  --experiment-sha256 "<64 lowercase hex from research --audit>" \
  --artifact /private/new-news-model-candidate.private.json

bash scripts/news-learning-review.sh candidate --verify \
  --cohort /private/news-cohort.json \
  --holdout /private/news-holdout-with-human-review.json \
  --review /private/news-human-review.json \
  --journal "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-research.sqlite3" \
  --artifact /private/new-news-model-candidate.private.json
```

Export only accepts an experiment fingerprint already present in the isolated
SQLite journal, recreates the TRAIN weights from pinned news and human review
files, and requires its validation Log Loss and baseline to match the recorded
run. Artifacts are JSON with a deterministic canonical payload SHA-256,
class token frequencies, document counts, algorithm version, research
experiment SHA, cohort/plan/review SHA and explicit
`productionEligible=false`. Output is exclusive create-only with owner-only
permissions and a bounded size. Export refuses missing/altered provenance.

Verification reloads the private JSON, validates schema and SHA, rebuilds
the same model from original TRAIN records and compares **every learned
parameter and every VALIDATION prediction exactly** to ensure restart replay.
The same dataset with new or later human reviews does not silently satisfy
the previous fingerprint. Verification does not evaluate TEST or promote a
model.

**Privacy:** token frequencies are derived from copyrighted/private source
headlines and summary text. Keep model JSON in the user's private local
research directory, never commit it to GitHub or upload to public CI.
A SHA-256 value detects modifications under a trusted recorded fingerprint;
it is *not* a digital signature, authenticity guarantee, or independent
assessment that human review is correct.

No weights are automatically exported by the research loop; a candidate must
first be journaled and then explicitly exported. Real human-gold source
quality, final independent holdout, shadow deployment, model selection,
rollback, 24/7 Mac launchd monitoring, and improved prediction metrics remain
open research/operations requirements. There is **no** stock-price, OHLCV,
trading-volume or realized-return data in this workflow.
