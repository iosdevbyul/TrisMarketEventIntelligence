# PR #274 — Immutable Gemma forecast commitments → actual future daily market returns

## Scope and real context

This is a Stage 2/3 market-result integration feature, **not** a new
classification audit. The automatic historical Gemma run had grown to
321 impacts, and the existing historical price set lacked returns
after the source announcement. PR #273 added **opt-in** local immutable
forecast commitments at the moment a *new* Gemma analysis is saved.
No operator Mac invocation of that opt-in has yet been supplied, so
there is **no real sealed forecast cohort currently established**.

This PR implements the next full read-only bridge: locally committed
model forecast → original exact body/answer/cutoff checks → externally
provided actual adjusted daily stock/benchmark price series → separate
1- and 5-trading-session absolute and relative returns → explicit
correct/incorrect/abstained/unavailable verdicts.

### Input sources

- Existing `news.sqlite3` with `automatic_forecast_commitments`,
  `automatic_news_analyses`, `articles` and `article_full_text`.
- A real, unchanged, original provider adjusted-daily price CSV in
  `symbol,date,adjusted_close,source` schema. A supplied price manifest
  may assert SHA equality, **not** third-party price authenticity.

The evaluator is read-only. It does **not** call Gemma4, collect
news, retrieve quotes, mutate production DB or allow manual edits
to stored commitments.

### Safeguards enforced BEFORE any return is scored

1. Validate the entire original local SHA chain, preserving each
   originally stored horizon/target/model/SHA. Altered local source
   commitments fail closed; local chain alone is not tamper-proof.
2. Match the immutable commitment to the original completed model
   record by article ID, original body SHA and model. Require the
   persisted original analysis generation time to match the commitment,
   plus unchanged source publisher/detection/body-observed clocks.
3. Rehash the unchanged source body and verify the committed
   **verbatim source evidence quote** remains inside that body.
   Confirm original target, direction and mechanism match the saved
   model response at the same impact index.
4. Only precommitted `CANDIDATE_ONLY_UNAUTHENTICATED_SOURCE_CLOCK`
   source/mapping eligibility passes. Invalid clocks, postdated
   mapping, late analysis and unmapped targets remain visible in the
   result with **no fabricated price verdict**.
5. Require valid exchange-local calendars for the originally sealed
   `XNAS`, `XNYS`, `XLON` or `XKRX` market. Use its exchange
   timezone, not Asia/Seoul for US/UK securities.
6. Original conservative daily study policy: first real daily close
   on a market date strictly **after** local source publication date
   is the entry, and the actual `horizon_sessions`-th subsequent
   trading close is the exit. Also reject a local forecast seal
   issued on/after the entry date. Nothing is backfilled with
   weekend, holiday, same-day or unobserved future prices.
7. Benchmark entry and exit must both have a real closing price
   on the same respective dates. Missing history is `UNAVAILABLE`,
   never an INCORRECT model prediction.
8. Compute absolute stock return, benchmark return, excess return
   and both direction verdicts at 0.1% neutral threshold, preserving
   original `UNCERTAIN` as abstention.

Each row reports the original news, original model and commitment
SHA, precommitted target/ticker/benchmark and market, recorded cutoff
clocks, exact price entry/exit dates and observation/eligibility
status. Per-horizon summary has local candidate agreement among
eligible non-abstained scored impacts, but **does not expose or
claim independently authenticated prospective forecast accuracy**.

### Critical limitations

The on-device SHA-chain and `sealed_at` timestamp are not independently
trusted or externally anchored. No independent earliest-public-source
attestation was supplied, and price provenance from Yahoo's adjusted
daily snapshots is not an exchange-signed point-in-time archive.
Consequently **`prospectivePredictionAccuracy=null` and
`trainingEligible=false`**, even when a hypothetical future row
shows local candidate CORRECT. It is descriptive conditional agreement,
not certified out-of-sample improvement or identified news causality.

No old historic 321 impacts are reclassified as prospective.
Never combine post-hoc historical #265/#269 results with the separate
new opt-in commitment cohort.

### Operator Mac commands

To enable **new-model-only** local commitments, user consent is
required because it modifies the live SQLite ledger. Only run a
bounded one-shot pass when desired (no watch, no background task):

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"

PYTHONPATH="$PWD/src" python -m \
  tris_market_event_intelligence.automatic_news_pipeline \
  --db "$DATA/news.sqlite3" \
  --model gemma4:e2b \
  --skip-collection --limit 10 --verification-runs 1 \
  --seal-forecasts \
  --mappings "$DATA/real-listed-equity-pr270-v1/reviewed-mappings.csv"
```

Inspect the local chain without modifying it:

```bash
PYTHONPATH="$PWD/src" python -m \
  tris_market_event_intelligence.automatic_forecast_precommitment_audit_cli \
  --db "$DATA/news.sqlite3"
```

Then evaluate ONLY saved local commitments against the existing
**actual** priced instrument history, without fetching more quotes:

```bash
PR270="$DATA/real-listed-equity-pr270-v1"
PYTHONPATH="$PWD/src" python -m \
  tris_market_event_intelligence.committed_forecast_market_outcomes_cli \
  --db "$DATA/news.sqlite3" \
  --prices "$PR270/expanded-adjusted-daily.csv" \
  --price-manifest "$PR270/expanded-adjusted-daily.csv.manifest.json" \
  --output "$DATA/commitment-market-outcomes-pr274-v1.json"
```

If no commitments have been opted in yet, returns
`status=NOT_OPTED_IN, committedRows=0`. If a commitment is present
but real future closes do not exist, it emits
`NO_POST_PUBLICATION_ENTRY_CLOSE` or
`INSUFFICIENT_LATER_SESSIONS`, *not* a claimed 0% accuracy.

The next acceptance gate requires verified earliest-public clocks,
auditable independent external time attestation, actual later
completed minute/daily price releases, and disjoint holdout
comparison against unchanged baseline decisions.
