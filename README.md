# TrisMarketEventIntelligence

Independent, local-first news-to-market evaluation and sequential learning system.

## Current goal

Collect historical news; analyze **one article at a time** with the configured AI;
identify exposed stocks, multiple sectors and broad-market proxies; seal each
prediction before looking at realized returns; fetch historical market prices;
compare predicted and realized price directions; diagnose failed predictions;
pass the resulting corrective lesson into the **next** article. Repeat up to
1,000 times, persist progress and resume after interruption. No dependency on
DonghakStockVision.

This is retrospective research, not investment advice or a live trading system.
A news/price correlation does not establish that the news caused the movement.
A model trained on later information cannot be claimed to pass a true historical
point-in-time evaluation without additional independent safeguards.

## Sequential study command

Use the existing locally collected `news.sqlite3`, installed `gemma4:e2b`,
and internet access for public Yahoo Finance chart quotes:

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
python -m tris_market_event_intelligence.sequential_market_learning_cli \
  --source-db "$DATA/news.sqlite3" \
  --as-of "2026-11-19T00:00:00+09:00" \
  --count 1000 \
  --model gemma4:e2b \
  --interval 1d \
  --horizon-hours 24 \
  --journal "$DATA/sequential-learning-v1.sqlite3"
```

The journal is **create/resume/extend**: the same run can increase `--count`
from 10 to 100 to 1,000 without reprocessing completed articles, provided
all other parameters remain unchanged. It never changes the source news DB.
Article predictions are committed before price evaluation. After each confirmed
mistake, the AI writes a diagnosis and a corrective instruction for subsequent
articles. Original errors remain immutable. There are no trade actions.

The research corpus is drawn from the checksum-verified existing news archive.
Articles are processed **one by one**, even when several articles were captured
on the same date. Exact duplicate content is excluded. The last 20% of dated
observations remain outside the training loop as a chronological holdout.
A smaller archive reports `INSUFFICIENT_CHRONOLOGICAL_ARTICLES` rather than
fabricating the missing 1,000 samples.

**Important:** this is a *retrospective adaptation experiment*, not a clean
point-in-time backtest. Learning from a 24-hour price outcome before processing
another article captured the same day uses information that was not available
at the second article's historical publication time. Such results MUST NOT be
represented as leakage-free historical prediction accuracy; a separately run
holdout evaluation is necessary. The original predictions remain immutable.

Current Yahoo Finance access is best effort and daily candles do **not**
establish minute-level causality. Missing quotes produce `UNVERIFIED`, never
an assumed correct answer or fabricated lesson. Research-generated lessons
are proposed prompt adaptations, not certified improvements.

## Article-by-article prediction audit (read-only)

Do not judge performance from `{"correct":0,"wrong":4}` alone. The
article audit shows what Gemma4 *actually predicted*, the source headline,
affected ticker or sector, original rationale, price proxy, observed direction,
absolute and benchmark-relative returns, baseline and outcome timestamps,
the stored verdict, diagnosis and next corrective rule. It also records which
memory rules influenced each prediction when that data was saved.

Run this against the **existing journal without re-running the LLM**:

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
python -m tris_market_event_intelligence.sequential_article_audit_cli \
  --journal "$DATA/sequential-learning-test-v1.sqlite3" \
  --source-db "$DATA/news.sqlite3" \
  --limit 1000 \
  --format html \
  --output "$DATA/sequential-article-audit-v1.html"
open "$DATA/sequential-article-audit-v1.html"
```

The standalone HTML report provides a searchable, expandable record for each
article. For Markdown use `--format markdown --output "$DATA/sequential-article-audit-v1.md"`.
For JSON consumers use `--format json --output "$DATA/sequential-article-audit-v1.json"`.
Output files are created exclusively; use a fresh filename for subsequent
exports. No source news or learning journal data are modified.

**Critical:** this reports historical *stored verdicts*, not certified
ground truth. Daily bars and the current +/-0.1% directional threshold must
be reviewed. Identical outcome timestamps across many articles, neutral
predictions marked WRONG, and absent asset prices are surfaced rather than
silently labeled learning success. A zero-target article is not a correct
prediction. Existing sealed predictions are never modified by the audit.

## Persistent learning memory (PR #252)

The sequential runner now persists long-term rule candidates in the same
`--journal` SQLite database. It upgrades existing completed article feedback
in place; it does not rewrite archived predictions or delete previous lessons.

Each failed prediction is categorized by event family and affected sector.
Repeated similar suggestions are combined instead of creating infinitely many
duplicate prompts. A new article retrieves relevant **ACTIVE** rules and one
exploratory CANDIDATE/VALIDATING rule rather than simply the latest 12 records.

New candidate rules are tested via a paired, sealed comparison on later articles:
the primary prediction receives the candidate; a shadow prediction omits it;
both predictions are generated before any prices are fetched. Only identical
resolved targets with public-information chronology eligible for that candidate
count as evidence. After at least 5 comparable future articles with at least
2 wins and more wins than losses, a candidate becomes ACTIVE. Clearly
underperforming candidates are RETIRED. Overlapping same-day outcomes may still
guide retrospective exploration, but **never** count toward promotion. The
chronological holdout remains untouched; no independent holdout score is
produced by this runner.

Read the durable state, failure families, article-to-rule applications and
paired trials without running Gemma or downloading any prices:

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
market-event-learning-report \
  --journal "$DATA/sequential-learning-test-v1.sqlite3"
```

Memory statuses mean CANDIDATE (hypothesis), VALIDATING (eligible paired
comparisons accumulating), ACTIVE (measurable paired improvement on later cases),
or RETIRED (insufficient or harmful paired evidence). ACTIVE is not a claim of
causation, trading utility, or an independently certified forward backtest.
A `verifiedImprovement` flag in the research report indicates ACTIVE rules
according to the above internal paired criterion, **not** independently
certified out-of-sample performance.

## Prior architecture

The following legacy documentation records earlier stages and is superseded
where it excludes market price evaluation from this project's scope.

## Local setup

Requires Python 3.11 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
ruff check .
pytest -q
market-event-api
```

Visit `http://127.0.0.1:8100/health` to confirm the service is running.

Optional settings: `MARKET_EVENT_HOST` (loopback-only) and `MARKET_EVENT_PORT` (default 8100).

## Delivery roadmap

1. **Foundation:** installable package, CI, isolated FastAPI health endpoint.
2. **Collector:** official-source adapters, source preservation, SQLite idempotency and retry.
3. **Event verification:** differentiate confirmed decisions from forecasts or commentary.
4. **News pressure analysis:** conditional direction, qualitative strength, evidence, uncertainty and model-provider adapter; no price predictions.
5. **Event API:** versioned event schema and cursor-based consumer polling.
6. **Mac operations:** launchd, observability, reboot and failure recovery.
7. **DonghakStockVision integration:** consume verified events without rewriting existing KRX pipelines.
8. **v1.0 verification:** live-source end-to-end, historical leakage checks and release criteria.

No real-world news collection or AI analysis is implemented in the foundation phase. The health endpoint alone is not a production-readiness signal.

## Project responsibility boundary (Stage 3 revised)\n\nSee [docs/news-impact-ownership.md](docs/news-impact-ownership.md) for the ownership contract and migration notes. The first additive `news_impact_signal_v1` schema accepts directional strength (`LOW`, `MODERATE`, `HIGH`, or `UNASSESSED`), an optional **news pressure index from 0 to 100, not a price percentage**, and provenance. Existing `market-impact-v8` records are adapted with `UNASSESSED` rather than inventing a strength score. Producing genuinely evidence-supported strength estimates is a separate model upgrade and evaluation task. Legacy minute-price study utilities remain for compatibility but are excluded from this module's future product scope.\n

### Pre-event market expectations and surprise assessment (sequential study)

New sequential predictions capture the evidence available **before** each article's
`firstObservedAt` timestamp. The contextual evidence contains up to five earlier
related articles (keyword-overlap candidates, not verified duplicates), a conservative
earlier SPY daily-close trend, and an explicit UNKNOWN consensus status. The
Gemma response may separately describe the economic impact, surprise versus market
expectations, novelty, possible prior pricing, and the expected price direction.
When no verifiable consensus source is connected, the surprise classification must
remain UNKNOWN; an earlier price increase alone never proves that an event was
priced in. No completed prior prediction is overwritten.

The HTML audit groups STOCK, SECTOR and MARKET targets separately and now shows
this evidence beside the original prediction, reasons and price outcomes. Legacy
entries without a surprise assessment remain visibly unavailable. These fields are
*analytical hypotheses* rather than verified market expectations or causal evidence.

The sequential feedback process remains **retrospective** and is not a leakage-free
forward-test benchmark. The proposed A/B/C performance comparison (article-only
vs pre-event context vs context plus learned rules) requires distinct controlled
runs and chronological held-out scoring; no such performance gain is claimed here.


### Historical news to listed asset mapping (sequential 100-article check)

The sequential market study compares prices against the **original reported
publication time**, not the RSS ingestion time. It uses the close before release
and the first session close at or after the release (the following session for
after-close releases). Exact intraday execution-price attribution is NOT
claimed; release times and daily bars still require source validation.

Before price scoring, stock predictions pass through an explicit
company-to-ticker consistency check for covered entities (including NVIDIA
NVDA, SK hynix 000660.KS and Samsung Electronics 005930.KS). Known mismatches
are rejected and recorded, never silently translated. The price provider's
actual returned symbol, exchange and currency must also agree with the
requested market; metadata mismatches produce UNVERIFIED rather than a wrong
or correct prediction. Supported U.S. and Korean ticker formats are
intentionally narrow. The known entity mapping is NOT a complete security
master: unrecognized firms still need better issuer identifier verification.
Sector proxies such as SMH or XLK represent U.S.-listed ETFs and must not be
confused with constituent stocks.

For a new controlled 100-article run, use a fresh journal path rather than
resuming a journal created before these asset checks. Model-format failures
are persisted as explicit SKIPPED_MODEL_ERROR entries rather than counted as
verified predictions. No independent forecast accuracy is claimed until the
report has been reviewed.

### Evidence-aware Korean-market rerun

The previous 100-case study scored Korean Bank of Korea clarifications against
U.S. SPY/XLF and often learned from same-day U.S. moves even when the article
body said only "see attachment". **Those legacy CORRECT/WRONG outcomes are not
a verified measure of article-driven price predictability.** They are preserved,
not overwritten. A new journal is required to use this research contract.

The revised sequential engine:

- Checks substantive source content first. Articles with a missing attachment,
  placeholder, or title-only body are **excluded from inference**, not counted
  as either neutral guesses or unverified price predictions.
- Supports an optional, immutable official-source PDF cache. Unsupported HWP,
  JS-only, unavailable or non-PDF attachments remain explicit unavailable
  records. The original RSS news database stays read-only.
- Uses Korea-listed KODEX 200 `069500.KS` as a broad Korean market proxy and
  KODEX Bank `091170.KS` as a Korean financial-sector proxy. These proxy
  choices do not imply article-specific causality; Yahoo metadata, exchange and
  currency must verify before any observed price move is scored. Other Korean
  sector proxies **abstain** unless a reviewed mapping exists. U.S. securities
  retain U.S. benchmarks. All symbols and their historical listing periods
  still require live quote-provider validation.
- Removes HTML-markup tokens from related article matching. Applies
  exchange-local trading session closes to 1-day price returns.
- Separates economic interpretation from scored asset prices. Whole-market
  daily moves are not causal evidence for a single release; only substantive
  articles with a meaningful sector/stock relative move can form provisional
  feedback hypotheses. Those hypotheses are never proof of causation.
- Makes past feedback available to a later historical article **only if the
  observed price outcome had occurred by that later publication timestamp**.
  Unverified chronology and retrospective documents remain explicitly
  non-certified PIT evidence.

If an official PDF is directly linked, collect its content into a new
read-only cache file before the sequential run:

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
python -m tris_market_event_intelligence.source_enrichment_cli \
  --source-db "$DATA/news.sqlite3" \
  --output "$DATA/official-content-100-v1.json" \
  --as-of "2026-11-19T00:00:00+09:00" \
  --limit 100

python -m tris_market_event_intelligence.sequential_market_learning_cli \
  --source-db "$DATA/news.sqlite3" \
  --content-cache "$DATA/official-content-100-v1.json" \
  --journal "$DATA/market-evidence-v2-100.sqlite3" \
  --as-of "2026-11-19T00:00:00+09:00" \
  --count 100 --model gemma4:e2b --horizon-hours 24 --interval 1d
```

The optional cache reports the number of PDFs actually extracted. If none are
available, the run processes only other *substantive* archived articles; it
must not silently backfill made-up content. No 100-article coverage guarantee.
A PDF retrieved in 2026 may have been replaced since its original publication;
therefore enrichment is retrospective and **not PIT-certified**. Daily candles
still cannot measure an exact within-session post-release response. Never
present a one-day directional coincidence as evidence that the original news
caused the price change.

### Evidence-grounded news-impact research

The event-grounding contract separates an allegation quoted in a Bank of Korea
press correction from the actual official response. Contradictions and clearly
administrative/ceremonial releases are abstentions, not "neutral price"
forecasts. Unsupported target mappings are rejected, and a listed ETF typed as
a STOCK is reconciled to its known MARKET or SECTOR proxy.

Evaluation reports keep three distinct concepts: observed absolute close-to-
close price movement, same-market benchmark-relative price movement, and
**unproven article-level attribution**. A neutral economic-impact assessment
does not assert the ETF must finish within a 0.1% daily price band. Observed
prices can still move for unrelated reasons. Post-hoc critic suggestions are
recorded only as unverified hypotheses and do not automatically become active
causal rules.

The sequential research runner now accepts `--cohort-policy diverse` (default)
or `chronological`. Diverse mode balances sources **only if actual source
variety exists**; it does not generate fictitious news and all selected articles
are replayed in historical publication order. Never reuse an existing journal
when changing any study contract.

For a distinct, no-feedback evaluation of reserved *later* articles after a
research run, execute:

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
python -m tris_market_event_intelligence.sequential_holdout_cli \
  --journal "$DATA/market-evidence-grounded-100-v1.sqlite3" \
  --source-db "$DATA/news.sqlite3" \
  --output "$DATA/market-evidence-grounded-holdout-v1.json" \
  --count 100
```

The holdout command never mutates the training journal, never produces
feedback or rules, and prevents overwriting its output. It reports the
directional-price denominator separately from neutral-event hypotheses.
A retrospective holdout using later-downloaded documents is **not
point-in-time-certified**, and when it supplies no learned rule context it
measures a frozen baseline rather than proven training improvement.

### Past-only market baseline + news-pressure A/B/C/D study

The original 83-event November 2026 holdout had only 17.73% recorded price-direction
agreement, compared with 85.82% for the trivial constant-negative control
during that short bearish window. This is neither a causal price model nor
evidence that the model is permanently inferior: the narrow period and market
regime make the sample unrepresentative.

A frozen holdout now records four **separate, pre-outcome decisions** on the same
observed targets:

1. Gemma4 price-pressure direction (existing research contract).
2. A deterministic **past-only price trend** from three completed exchange-local
   trading sessions before the article's published timestamp. The Yahoo query
   ends at publication time; the bar with a session close after release is never
   a feature.
3. A prespecified signed sum of price-trend and Gemma4 news pressure. Neutral
   news pressure means *no adjustment*, not a prediction that tomorrow's asset
   price will be flat.
4. An optional **training-journal calibration rule**, fitted once from earlier
   complete price observations, with a minimum sample of eight per
   market-and-predicted-direction bucket. If historical support is insufficient,
   the trained rule abstains; it cannot use held-out outcomes or assume a
   Korean ETF historical label generalizes to U.S. securities.

The report keeps paired denominators for baseline, combination and trained
calibration, rather than ranking accuracies computed on different subsets. It
also exposes source/month distributions, unchanged constant-direction
controls, price-direction labels versus same-market relative direction,
and explicit `causalAttributionEstablished: false`. The calibration learns
**past price coincidence**, not whether a specific news event caused that
move. This does not establish a successful AI learning gain. If the selected
holdout spans only one month, the output explicitly reports insufficient
regime diversity.

Use the already completed **original** 100-case journal, not a new training run.
The command below only creates one new holdout output; it does not modify
the source archive or the training journal:

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"

python -m tris_market_event_intelligence.sequential_holdout_cli \
  --journal "$DATA/market-evidence-v3-100.sqlite3" \
  --source-db "$DATA/news.sqlite3" \
  --output "$DATA/market-holdout-baseline-news-v2.json" \
  --count 83
```

Running this study again uses Yahoo historical price queries and Gemma4
inference for each eligible holdout article. Verify the source dates and
historical content separately; publication timestamps, retrospectively
downloaded PDF documents and third-party daily prices are not
point-in-time-certified. Do not merge or enable automatic learning until
the independent source regime, price alignment and multi-period robustness
checks have been reviewed.

### Thirty genuinely different release days (frozen exploratory test)

A chronological holdout that contains 83 articles collected from only three
calendar days **cannot** establish 30 independent article dates, however many
articles it has. The new `spread-dates` sampling contract examines all
eligible **untrained, post-training-release** source records (not just the
existing trailing 20% cohort) and excludes the article IDs in the old
83-article holdout. It selects at most one article for each local trading
market release day, spanning the available period using a deterministic
maximin-date algorithm. The selection **never** consults price outcomes.

First inventory real available candidates on the Mac using the read-only
original news archive and original 100-article training journal:

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
python -m tris_market_event_intelligence.sequential_holdout_cli \
  --journal "$DATA/market-evidence-v3-100.sqlite3" \
  --source-db "$DATA/news.sqlite3" \
  --exclude-report "$DATA/market-holdout-baseline-news-v2.json" \
  --selection-policy spread-dates --count 30 \
  --preview --manifest "$DATA/research-30-distinct-dates-manifest-v1.json"
```

The immutable manifest shows original article IDs, full source URLs, original
reported publication timestamp, market-local date, title, date span, source
composition and whether 30 distinct dates really exist. If the archive
contains fewer than 30 independent dates, **do not** run or report a
30-date evaluation; ingest older real releases first, with their verifiable
original publication times, and repeat the preflight. A failed preflight does
not establish that 30 historical releases exist.

Once `dateSpread.sufficientDistinctDates` is true and the manifest has been
reviewed, execute the exact same selection against the same source DB:

```bash
python -m tris_market_event_intelligence.sequential_holdout_cli \
  --journal "$DATA/market-evidence-v3-100.sqlite3" \
  --source-db "$DATA/news.sqlite3" \
  --exclude-report "$DATA/market-holdout-baseline-news-v2.json" \
  --selection-policy spread-dates --count 30 \
  --output "$DATA/research-30-distinct-dates-results-v1.json"
```

The study fails before Gemma inference or outcome queries if fewer than 30
dates are eligible, and existing report files are never overwritten.
Dates are distinct by jurisdiction-local release date, not by RSS receipt
time. Results are explicitly labeled
`UNTRAINED_POST_TRAINING_ARCHIVE_NOT_ORIGINAL_HOLDOUT`: this is a
retrospective exploratory sample, **not** the untouched strict last-20%
holdout and not verified source-PIT. The training DB is read-only and no
feedback rules are applied. Multiple distinct dates do not by themselves
establish market regime independence; inspect per-date returns and compare
against constant-direction and past-only trend baselines.

### Economic-event-first research cohort: 30 distinct dates

The preceding 30-day exploratory experiment selected articles using release
dates rather than economic-event suitability. It found 30 different dates
across April–November 2026, but only 18 articles had priced targets (23 target
observations). The price-direction coincidence rates were Gemma-only 2/23,
past-price 8/23, naive combination 7/23, and evidence-gated 8/23 (no gating
overrides). Those results do not prove causation or broad independent
generalization. Editorial/programme titles and reports with missing original
content must not fill a future economic-impact research cohort.

The new `event-spread-dates` policy **screens before selecting dates**.
Only explicit monetary-policy actions, real statistical releases, earnings or
guidance changes, contracts/corporate actions and regulatory decisions qualify.
It rejects programmes, speculative editorials, purely administrative
clarifications, title-only/attachment-placeholder articles, and missing
source URLs. Selection never consults actual price returns, prior correctness
or model predictions. It records event categories, exclusion reason counts,
available independent release dates and complete article manifest entries.
Qualifying candidates are not yet certified to have moved market prices or to
have clean source point-in-time provenance.

**First inspect the real archive; it may have fewer than 30 qualifying dates.**
The exclusion options intentionally cover BOTH prior 83-event and 30-event
research results. This produces an immutable manifest without Gemma/price
queries or mutation of the original news/training stores:

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"

python -m tris_market_event_intelligence.sequential_holdout_cli \
  --journal "$DATA/market-evidence-v3-100.sqlite3" \
  --source-db "$DATA/news.sqlite3" \
  --exclude-report "$DATA/market-holdout-baseline-news-v2.json" \
  --exclude-report "$DATA/research-30-distinct-dates-results-v1.json" \
  --selection-policy event-spread-dates \
  --count 30 --preview \
  --manifest "$DATA/market-event-research-30-manifest-v1.json"
```

Inspect `economicEventScreen`, `dateSpread.items` and
`dateSpread.sufficientDistinctDates` before proceeding. If fewer than 30
qualifying dates exist, **do not lower the quality threshold just to reach
30**; acquire more correctly timestamped original economic announcements,
earnings, contracts and regulatory releases, then repeat the preflight.
With 30 verified eligible dates in the saved manifest, run the exact same
sample with a manifest consistency check:

```bash
python -m tris_market_event_intelligence.sequential_holdout_cli \
  --journal "$DATA/market-evidence-v3-100.sqlite3" \
  --source-db "$DATA/news.sqlite3" \
  --exclude-report "$DATA/market-holdout-baseline-news-v2.json" \
  --exclude-report "$DATA/research-30-distinct-dates-results-v1.json" \
  --selection-policy event-spread-dates --count 30 \
  --selection-manifest "$DATA/market-event-research-30-manifest-v1.json" \
  --output "$DATA/market-event-research-30-results-v1.json"
```

Running against a changed article set, content cache or selection policy
fails before inference. The training journal remains immutable. The model
does not receive prior feedback lessons on holdout. Economic indicators
may still lack a reliable pre-release consensus, individual security
exposure and independently timestamp-certified full text; the engine
records such limits and must not claim a causal forecasting success.

### Opt-in SEC company 8-K backfill for real earnings and contract events

The existing 30-date event-first cohort only had **27 eligible days** after
excluding prior studies, all from the Bank of Korea. Do not pad the sample
with editorials or duplicated events. A new opt-in official source loader
uses SEC issuer submission history and downloads **actual primary Form 8-K
filing documents**, restricting events to Item 2.02 (reported earnings/
results) and Item 1.01 (material definitive agreements).

The importer uses the existing `articles` / `captures` SQLite tables,
persists exact original filing bytes and hashes, rejects broken or missing
documents, and deduplicates by official accession. It does not create events
from API metadata alone. SEC submission API timestamps can disagree with
the official filing-detail Accepted clock: before importing each filing the
loader cross-checks the detail page's US-Eastern Accepted timestamp, then
uses that corroborated instant for research chronology. **SEC filing acceptance
is still not proof of the earliest issuer press release.** For market-price
study, review first-publication evidence and distinguish the post-acceptance
price response from a stock move that may already have happened.

SEC requests must provide a real identifying User-Agent as described in
SEC's developer guidance. Set `SEC_USER_AGENT` locally with your own
application/contact information; never commit contact details to Git. No
automatic crawling, background scheduler or trading is enabled.

First preview real SEC candidate counts (read-only), for issuer CIKs
NVIDIA 1045810, AMD 2488, Apple 320193, Microsoft 789019, Alphabet
1652044 and Meta 1326801:

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
export SEC_USER_AGENT="YourResearchApp real-contact@example.com"

python -m tris_market_event_intelligence.sec_issuer_event_acquisition_cli \
  --db "$DATA/news.sqlite3" \
  --ciks 1045810,2488,320193,789019,1652044,1326801 \
  --since 2026-04-28 --until 2026-11-18 \
  --output "$DATA/sec-issuer-8k-preview-v1.json"
```

Inspect the candidate count, acceptance clocks and CIK/filing URL mapping.
To opt in to importing only **real fetched primary document bytes** use
`--import-files` and a *different* output report path, e.g.
`$DATA/sec-issuer-8k-import-v1.json`. A missing response does not cause
fabricated content. Import may add new articles to the **existing** news
archive, but never alters the sealed training journal.

The 100-case training journal retains its historical article-inventory cutoff.
To select genuinely newly captured SEC articles, use the separate research
`--inventory-as-of` option, fixed to the actual post-acquisition UTC time.
Pass that **same cutoff value in both preview and actual evaluation**, and
pass `--selection-manifest` to prevent changed candidates. In particular
this is an expanded **retrospective exploratory** research sample, not the
original untouched holdout and not automatically a certified PIT benchmark.
Retrospective SEC filings may have been publicly available after an earlier
issuer announcement. Never treat filing acceptance as proof of original
market-information arrival.

The 30-date event gate should be applied after acquisition by rerunning
`--selection-policy event-spread-dates` with both previous research reports
excluded. If fewer than 30 distinct credible release dates survive, record
that shortage rather than claiming a successful 30-event benchmark.
