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

## Gemma4 structured-output correction (2026-11-15)

Actual Mac Gemma4:e2b returned two `MODEL_SCHEMA_INVALID` results with
the earlier unconstrained `format: "json"` generation. This confirms schema
validation failed for both responses; the exact missing fields were **not**
available in the original reports and content correctness is unknown.

The small `evidence-pilot` now sends Ollama a flat explicit JSON schema
with required keys, fixed enum values and three false-only flags. It still
performs the existing complete Pydantic validation and immutable original
news evidence checks afterward. The full-cohort `materiality-local` path
is unchanged (plain JSON mode). A rejected sample includes only a
whitelisted `schemaFailureField` and a `SCHEMA_FIELD_*` count, never the
model's private completion or source excerpts in terminal output.

After updating main, repeat the **same 2 candidates** using a new private
filename (do not replace the earlier evidence files):

```sh
bash scripts/news-learning-review.sh --update evidence-pilot --run \
  --cohort "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-cohort-20261115.json" \
  --sample "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-review-sample-20261115.json" \
  --model gemma4:e2b --limit 2 \
  --output "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-evidence-pilot-structured-20261115.json"
```

Ollama may reject unsupported schema grammars and could still return bad
outputs. This is evidence of actual runtime compatibility, not a hidden
automatic fallback or a claim that any hypothesis is factually accurate.

## Gemma4 root-level JSON parsing failure follow-up

The real Mac structured trial returned `MODEL_JSON_INVALID: 2` with
`SCHEMA_FIELD_ROOT: 2`. This is a *parse* failure, not an identified missing
field or evidence mismatch. The earlier reports omit Ollama completion metadata,
so an output limit, empty answer and malformed complete JSON cannot be
distinguished retroactively.

The opt-in small pilot now asks Ollama to disable thinking (`think: false`)
and allocates a bounded **2048 output tokens** only in structured pilot mode.
Other full-cohort production/research routes keep their original settings.
It checks `done_reason: length` and empty content before the Pydantic parser.
Explicit codes `MODEL_GENERATION_TRUNCATED`, `MODEL_EMPTY_CONTENT`, and
`MODEL_JSON_INVALID` are distinct. No automatic second attempt, no silent
text repair, no relaxing evidence checks or inventing human gold.

Use a **new** create-only private output for the next single small Mac check:
```sh
bash scripts/news-learning-review.sh --update evidence-pilot --run \
  --cohort "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-cohort-20261115.json" \
  --sample "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-review-sample-20261115.json" \
  --model gemma4:e2b --limit 2 \
  --output "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-evidence-pilot-completion-20261115.json"
```
If the installed Ollama build does not accept `think: false`, it will fail
closed with an HTTP status error. We cannot claim support until a real Mac run.

## Learning-05 — expand the pinned 48-candidate window safely

The original 2 news hypotheses (sample indexes 0 and 1; article IDs 336,
410) have already been generated and reviewed. Do **not** re-run those
same candidates when expanding coverage.

The existing opt-in `evidence-pilot` can now process a *contiguous*
zero-based window of the immutable deterministic 48-candidate packet.
The previous `--limit 2` default and `--start-index 0` default remain
unchanged for compatibility. At most 12 selected news articles can be
sent per explicit command. The range must fit entirely within the pinned
sample; all original 482-article cohort and sample integrity checks occur
before the first local model call.

A first extension is the six **new** candidates at indexes 2 through 7:

```sh
bash scripts/news-learning-review.sh --update evidence-pilot --preview \
  --cohort "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-cohort-20261115.json" \
  --sample "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-review-sample-20261115.json" \
  --model gemma4:e2b --start-index 2 --limit 6
```

To actually run the same six items only after choosing to execute locally:

```sh
bash scripts/news-learning-review.sh evidence-pilot --run \
  --cohort "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-cohort-20261115.json" \
  --sample "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-review-sample-20261115.json" \
  --model gemma4:e2b --start-index 2 --limit 6 \
  --output "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-evidence-pilot-candidates-02-07-20261115.json"
```

The report includes `sampleStartIndex`, `sampleEndIndexExclusive`,
and ordered `selectedArticleIds` so every subset is auditable. This
operation does not globally schedule, train, or promote the model and does
not modify a prior evidence file or source database. No retries/repairs
are invented for rejected cases. It is not automatically resumable or
a claim of full 48-candidate coverage: separate, explicitly initiated
non-overlapping windows are needed if the six-item analysis is acceptable.

Do not equate source-literal and schema contract acceptance with
human adjudication, semantic accuracy, or evidence of LLM improvement.
The original sample still spans too few UTC observation days for
independent generalization claims. Its human gold remains absent, TEST
sealed, and all stock prices/returns/OHLCV outside the project.
