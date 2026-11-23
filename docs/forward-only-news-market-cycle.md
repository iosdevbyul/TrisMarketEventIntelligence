# PR #276 — Real newly published news → at-generation forecast → actual market outcome

## Market mission alignment

The project charter defines the outcome as **actual subsequent stock / sector /
index / benchmark return versus a prior market-direction forecast**, with
separate historical and independently validated prospective samples. News
classification and model-written explanations support that goal.

Real local operator results so far were **321** retrospective Gemma impact
rows, plus historical Yahoo adjusted daily NVDA/SPY/CPI.L/ISF.L observations
through 2026-11-20. Newly published 2026-11-23 articles had no later close
within that snapshot; none could be directionally scored. The user has not
supplied evidence of any real at-generation opt-in forecast commitments after
PR #273, so **no actual prospective accuracy has been measured**.

PR #273 introduced opt-in local commitment sealing, PR #274 read-only
commitment → real-price evaluation, and PR #275 optional rescoring of
commitments alongside new verified daily price observations. The missing
real-world operation was **how to get NEW news into a forward-only cohort
without mixing the hundreds of historical articles awaiting Gemma processing**.

## Full new forward capture and evaluation path

The existing `automatic_news_pipeline` accepts opt-in
`--fresh-only-minutes N` **only together with**
`--seal-forecasts --mappings <approved.csv>`.

- Selects *previously unprocessed or retryable* original articles ordered by
  actual detection timestamp descending, not article ID, preventing the
  historical backlog from taking the limited candidate slots.
- Requires timezone-aware **publisher-stated publication**, detection, and
  first saved complete-body observation timestamps; verifies
  `fresh_cutoff <= published <= detected <= body_observed <= current`
  for the bounded window. Published-old/reposted stories, missing
  original-body observation, naive/malformed/future clocks and internally
  inconsistent captures are rejected **before invoking Gemma**.
- Revalidates original body digest, complete status and sparse page filters
  using the existing worker; model source input and output handling are not
  reinvented. Only truly new model results can create commitments.
- The normal historical pipeline **retains its prior backlog selection**.
  Neither old article rows nor completed model outputs are retroactively
  sealed by enabling this option.
- During a successful new model write, source quote, target, original UP /
  DOWN / NEUTRAL / UNCERTAIN forecast, market mapping and 1d/5d horizons
  are saved in the same DB transaction and validated against source and
  timecutoff in the existing PR #273 precommitment ledger.

New `forward_news_market_cycle_cli` provides a **single explicitly invoked,
bounded one-shot** operational command:

1. Optionally collect real RSS/source fulltext (default, or
   `--skip-collection` for already-fetched recent originals).
2. Analyze only those *genuinely fresh*, original, as-yet-unprocessed
   articles using the configured local Gemma model, with opt-in
   atomic 1d/5d commitment capture.
3. **After** the forecast and its source evidence are saved, read the
   user's existing real adjusted-price CSV and optional SHA manifest;
   perform the PR #274 read-only committed forecast market evaluation.
   Never include price data as model input.
4. Write one **create-only** JSON run report: actual article and source
   capture status, source-time eligibility and mapping exclusions,
   commitment chain summary, existing real market price observation
   and 1d/5d absolute/relative direction score/abstention/unavailable
   metrics. It never classifies still-unobserved prices as model losses.
5. Real later observed prices can subsequently be collected through
   PR #275's `incremental_daily_market_maturity_cli
   --include-committed-forecasts` with the original at-generation cohort
   unchanged. No additional model calls when scoring matured returns.

## Optional real Mac run (NO background loop)

From a checkout after this PR is merged:

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
PR270="$DATA/real-listed-equity-pr270-v1"

PYTHONPATH="$PWD/src" python -m \
  tris_market_event_intelligence.forward_news_market_cycle_cli \
  --db "$DATA/news.sqlite3" \
  --model gemma4:e2b \
  --mappings "$PR270/reviewed-mappings.csv" \
  --prices "$PR270/expanded-adjusted-daily.csv" \
  --price-manifest "$PR270/expanded-adjusted-daily.csv.manifest.json" \
  --fresh-minutes 120 \
  --limit 10 \
  --output "$DATA/new-forward-market-cycle-pr276-v1.json"
```

This deliberately **changes the live SQLite database** by appending only
truly NEW model analyses and commitments, and can access sources and
local Ollama. It is **not** an existing default or a silent activation;
the operator must explicitly run it. It does not download market prices
or run `--watch`, install daemons, trade or launch a 24/7 service.
Use `--skip-collection` to suppress any new RSS network collection if
only already-fetched recent news should be processed.

If no fresh originals are found, `newArticleAnalyses=0` and the existing
committed results may still be reviewed. This is normal, not a forecast
failure. A new output name is required for each one-shot run.

### Evidence boundary

This command cannot independently authenticate the first public
availability from a publisher's self-reported `publishedAt`, distinguish
all syndication families, independently sign the local wall-clock/SQLite
hash chain, obtain an exchange-signed point-in-time minute quote archive
or generate actual unobserved future prices. **A recently captured story
is not proof of a verified first-public release**; even locally
apparently admissible rows remain `CANDIDATE_ONLY_UNAUTHENTICATED_SOURCE_CLOCK`.
All summary reports continue to say
`prospectivePredictionAccuracy=null`, `trainingEligible=false`
until external evidence and independent event-family holdouts exist.

The Stage 2–3 value of this feature is **actually producing a new,
separate, source-time forecast population** and providing direct
observed-price outcome evaluation instead of endlessly relabeling old
historical Gemma analyses.
