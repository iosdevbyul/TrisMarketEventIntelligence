# Immutable opt-in new-model forecast commitments — Stage 2 foundation

## Why this is the next substantive market-learning feature

Actual local results: The automatic Gemma ledger expanded from **103 to 321
impact rows**, but **0** were directionally scoreable on the same historical
price file; existing NVDA and Capita announcements were made 2026-11-23
and the saved daily adjusted close data stopped at 2026-11-20.
The incremental source-price collector correctly returned zero new bars
when the 2026-11-24 06:54 KST invocation supplied 2026-11-23 UTC as the
**exclusive** end, which contains no new trading session.
Creating synthetic later bars or re-running the same range is not progress.

More importantly, the existing automatic worker persists Gemma answers
but does **not** make immutable market-target/horizon commitments at
generation time. Re-evaluating a past Gemma answer against prices can
NEVER be turned retroactively into a prospective forecast. This PR
fills the capture-side gap, not a verified trading performance gap.

## Implementation

New opt-in `--seal-forecasts --mappings <reviewed.csv>` options on
`automatic_news_pipeline`. For genuinely NEW successful Gemma
inferences only, and within the same SQLite transaction as the completed
model ledger entry, the pipeline creates one independent
`automatic_forecast_commitments` row per model impact and per
precommitted 1/5-trading-session horizon. Every original model target
and original UP/DOWN/NEUTRAL/UNCERTAIN decision is preserved, including
abstentions and unmapped targets; market mappings are NEVER guessed.

Each commitment records source ID, original body SHA, model, exact source
quote and reasoning, publisher-stated/detection/body-observed/model
processing clocks, exchange, *at-seal-time* reviewed ticker and
benchmark, mapping verification time, horizon/session policy and a
canonical SHA256 linked to its previous commitment record.
`BEFORE UPDATE` and `BEFORE DELETE` SQLite triggers reject mutation.
No prices or outcomes are consulted anywhere in commitment creation.

The conservative eligibility classifier marks any inaccurate/out-of-order
source clocks `UNVERIFIED_SOURCE_CLOCK`, missing precommitted symbol
`NO_PRECOMMITTED_INSTRUMENT`, postdated mapping verification
`MAPPING_NOT_VERIFIED_AT_COMMITMENT`, and processing after the
exchange-local publication date `AFTER_CONSERVATIVE_LOCAL_CUTOFF`.
Only records fitting this limited check get
`CANDIDATE_ONLY_UNAUTHENTICATED_SOURCE_CLOCK`.

**No record is approved as externally verified prospective**.
A machine-local SQLite clock and SHA chain are NOT an independent
timestamp authority or tamper-proof archive. No first verifiable
public availability is established by capture metadata; detection is
not automatically original first publication. Even an eligible
candidate must pass independent source/price publication-time evidence
before a prospective prediction-accuracy claim.

The separate `automatic_forecast_precommitment_audit_cli` is read-only
and checks the whole local hash chain, count and eligibility. A
locally altered hash chain or bogus future source clock causes a
failure, not recovered labels. The existing Gemma watch worker remains
**opt-in foreground**, with no automatic trading, background job,
provider price fetch or model learning.

## Usage

The existing foreground news worker can remain stopped. No historic
analysis is silently backfilled. To enable commitment capture for
**future newly analyzed** articles only, from main after this PR:

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"

PYTHONPATH="$PWD/src" python -m \
  tris_market_event_intelligence.automatic_news_pipeline \
  --db "$DATA/news.sqlite3" \
  --model gemma4:e2b \
  --skip-collection \
  --limit 10 \
  --verification-runs 1 \
  --seal-forecasts \
  --mappings "$DATA/real-listed-equity-pr270-v1/reviewed-mappings.csv"
```

To inspect the saved local commitments without any model calls:

```bash
PYTHONPATH="$PWD/src" python -m \
  tris_market_event_intelligence.automatic_forecast_precommitment_audit_cli \
  --db "$DATA/news.sqlite3"
```

The pipeline processes only unsolved new analyses; running once after
many articles have been analyzed without opt-in cannot retroactively
commit their original forecasts. The mapping file contains only
already reviewed names and does not magically resolve broad
macro/news-sector strings. Unmapped cases are preserved as
unscoreable, not automatically linked to any later chosen ticker.

A local commitment registry is **not** a prospectively verified
forecast evaluation itself. The next work must connect eligible
independently timestamp-attested, real-time candidate entries to
genuinely subsequent stock/benchmark quotes under strict PIT market
calendar rules. Separate distinct historical/research cohorts from
future truly precommitted external-time-authenticated predictions.
Do not declare model learning or directional superiority without
independent holdout versus matched trivial baselines.
