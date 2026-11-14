# Stage 4 — Read-only Donghak KRX market snapshot readiness

This adapter is the first integration boundary between the event study project
and DonghakStockVision's **existing** KRX market SQLite files. It does not
modify DonghakStockVision, copy operational databases, query KRX APIs, or
promote unqualified research data.

## Verified upstream schema contracts

Source: `iosdevbyul/DonghakStockVision` at implementation review:
- Research DB `bars(ticker, trading_date, payload)` where `payload`
  represents `DailyBar`. Metadata has `provider`, `market`, `adjustment`,
  `trading_date`, `close`, `volume`.
- Operational DB `market_observations(market, trading_date, ticker, payload,
  validation_status, ...)`, distinct application ID `0x4453564F` and
  `user_version=1`. Operational data is not automatically research-qualified.
- The existing KRX OHLCV endpoint is **unadjusted**. A raw observed KRX close
  is **not** adjusted close; substituting one silently would contaminate
  corporate-action historical returns.
- A KOSPI stock collection is **not** a historical KOSPI index series. An
  independent, trusted and time-aligned market benchmark is still required.

## Mac workflow

The previously established one-line offline verification remains unchanged:

```sh
bash scripts/verify.sh --update
```

It automatically runs all new read-only adapter regression tests under the
existing Intel Mac Conda/tmux compatibility path.

For a human-authorized **offline stable copy** of a research/operational DB,
the optional *readiness only* command is:

```sh
market-event-krx-readiness \
  --market-db /absolute/path/to/reviewed/offline-market.sqlite3 \
  --sha256 ACTUAL_64_CHARACTER_LOWERCASE_SHA256 \
  --tickers 005930,000660
```

You must obtain the exact DB and its SHA-256 from an existing approved
snapshot. Paths and hashes are **not** guessed by the CLI. Never pass live
`/Users/comatoki/KRX-Private` operational storage: prepare and approve an
offline snapshot using DonghakStockVision's existing procedures instead.
For operational snapshots, require a stable `.snapshot.sqlite3` file.

The adapter verifies the file hash and SQLite integrity, detects an active WAL
or journal, opens `mode=ro&immutable=1` with `query_only`, and checks the
fingerprint before/after each read. It cannot create or repair the source DB,
change authorization, or access API credentials. Results report only ticker
coverage and adjustment/eligibility statuses, not proprietary raw payloads.

Unadjusted, unknown adjustment, no-trade and unapproved operational bars are
**excluded** from the Event Study adjusted price boundary. A syntactically
adjusted research series can be read, but this is **not** independent proof
of underlying KRX authenticity, adjustment validity, missing-date absence,
point-in-time validity or event label readiness.

This is a fail-closed integration readiness step, **not yet a trained model or
market impact backtest**. The report deliberately sets
`benchmark_ready=false`, `event_study_dataset_ready=false`,
`training_ready=false`, `quality_verified=false`, because the benchmark,
corporate action quality review, historical completeness and event-to-ticker
bindings remain outstanding. Do not use this audit as trading advice.
