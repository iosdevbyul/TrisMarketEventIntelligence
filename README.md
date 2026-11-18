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

The journal is **create/resume**, and never changes the source news DB.
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