# PR #270: Direct named listed equities → actual adjusted daily price observations

## Verified real operator result and problem definition

On the real Mac the stored automatic Gemma ledger was evaluated in merged
PR #269. **63** original article analyses yielded **90** impact records:
**72 UNMAPPED, 15 NEWS_SOURCE_NOT_ADMITTED, 2 price-observed ABSTAINED, and
1 NO_ENTRY_SESSION**. Direction-scored denominator: **zero**.
`accuracy=null` is NOT 0% performance. The historical predictions were
retrospective and never sealed before an outcome.

The user examined all 67 distinct unmapped target strings and identified
two *direct named listed company* directional cases:

- **Article 20975, "Nvidia", UP**. <https://investor.nvidia.com/investor-resources/faqs/default.aspx>
  confirms the company common equity is Nasdaq `NVDA`. Use XNAS and
  SPY as an explicitly US-large-equity market benchmark.
- **Article 20989, "Capita", DOWN**. <https://www.londonstockexchange.com/en/quote/equity/CPI/capita-plc>
  confirms London equity `CPI`; Yahoo's London instrument is `CPI.L`.
  Use XLON, `Europe/London`, and `ISF.L` as a declared UK
  large-cap ETF comparison. BlackRock's
  <https://www.ishares.com/uk/individual/en/products/251795/ishares-core-ftse-100-ucits-etf-isf>
  establishes the FTSE 100 ETF. **The ETF may differ materially from
  Capita's business exposures**, and the comparison is descriptive.
  Capita's 23 November 2026 7 AM UK market announcement and large
  same-day price fall are publicly documented, but the exact operator
  article 20989's first-public information cutoff and timing are not
  independently established from its name alone.

Other examples are deliberately **excluded**. South East Water is
privately held; London Stock Exchange may refer to the exchange
trading venue or an index rather than London Stock Exchange Group Plc
equity; "AI hyperscalers", "Treasury securities", "AI equity prices",
and broad sectors are not a single named company share. It would be
invalid to give them arbitrary company/ETF returns solely to inflate
coverage.

## Real integration implemented

`listed_equity_price_coverage_cli` operates on the *existing immutable*
PR #269 actual automatic-ledger outcome JSON plus original reviewed mapping
and historical adjusted price CSV. Only records whose exact
`(target, articleId)` matches the two issuer candidates **and** whose
source audit matched the stored original body **and** predicted `UP` or
`DOWN` are selected.

This CLI:

1. Extends the **original preserved** target/market/benchmark mapping CSV
   with separate source-backed XNAS and XLON company share candidates.
   Listing evidence and *true current review time* are retained.
   Historical verification is **retrospective**.
2. Records the precise 4 quote symbols `NVDA`, `SPY`, `CPI.L`,
   `ISF.L`, and original quote availability.
3. If `--fetch-prices` is explicitly set, calls the existing yfinance
   `interval=1d`, `auto_adjust=True` collector for the specified date
   interval. These are **real provider observations**, not generated
   sample prices. Missing provider data remains unavailable.
4. Enforces `--end` **exclusive and no later than the current UTC
   calendar date**; this excludes the current incomplete trading-day
   close from reported historical daily returns.
5. Preserves the complete newly fetched price series per instrument and
   retains older historical source series for wholly absent instruments
   (never merges conflicting individual dates or fabricates gaps).
6. Creates SHA-pinned adjusted-price recovery manifest and reruns the
   **actual current** automatic-Gemma SQLite-to-market evaluation using
   original news integrity checks, 1-/5-session stock returns and
   benchmark-relative returns.
7. Reports direction-scored denominator *before* and *after*, completed
   observed impact count, provider missing instruments, quote row counts
   and exact outcome items. Historical source files are read-only and
   outputs are an entirely new immutable directory.

Potential outcome caveat: A 2026-11-23 UK issuer article cannot be
awarded a 1-/5-day return on that same date; neither today's incomplete
bar nor future sessions may be guessed. The existing conservative
daily study deliberately starts at the first close on a **strictly
later trading date** than source publication and measures subsequent
sessions. It does **not** capture the immediate 7 AM UK news gap and
does **not** establish event causality. Such intraday study requires
licensed original timestamped minute quotes and an independently
verified earliest source public-release time.

The CLI also supports plan-only mode (no network/provider or SQLite
writes) for diagnostic safety.

## Real Mac invocation

Execute only from a checkout containing the latest merged PR; the
current Mac must have the previously supplied *real* PR #269
outcome file and local automatic Gemma ledger.

As of **2026-11-23 UTC**, do not include that day's unfinished close.
The following command explicitly authorizes the optional Yahoo historical
price lookup; change the `--end` only once later daily bars are completed
and available:

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"

PYTHONPATH="$PWD/src" python -m \
  tris_market_event_intelligence.listed_equity_price_coverage_cli \
  --outcomes "$DATA/automatic-gemma-real-daily-outcomes-pr269-v1.json" \
  --db "$DATA/news.sqlite3" \
  --mappings "$DATA/stage6-sector-target-map-v2.csv" \
  --prices "$DATA/stage6-sector-prices-v4-recovered.csv" \
  --model gemma4:e2b \
  --start 2026-01-01 \
  --end 2026-11-23 \
  --fetch-prices \
  --output-dir "$DATA/real-listed-equity-pr270-v1"
```

The network fetch is optional. Remove `--fetch-prices` (and dates) to
generate *only* source-backed quote and target plans. Output directory
**must not exist** beforehand; the CLI refuses overwriting.

When successful, inspect:

- `coverage-plan.json`: new issuer candidates, price provider coverage,
  `scoredDirectionBefore`, `scoredDirectionAfter`, and
  `newDirectionScoresObserved`. A positive change proves additional
  historical price comparisons, **not** model improvement or prospective
  forecast accuracy.
- `automatic-observed-outcomes.json`: per-event absolute/relative
  direction/verdict and every remaining unavailable/unmapped/blocked case.
- `expanded-adjusted-daily.csv` and its `.manifest.json`: genuine
  price observations and source lineage.

No 24/7 autonomous monitoring, no automatic training, and no trading
are enabled. Everything remains `RETROSPECTIVE_UNSEALED`,
`prospectivePredictionAccuracy=null`, and `trainingEligible=false`.
