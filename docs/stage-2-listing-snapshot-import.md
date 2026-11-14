# Stage 2 — Source-bound KR/US issuer snapshot import (not historical listings)

The existing Stage 2 read-only issuer matcher accepts a tightly controlled
CSV. This step adds **an explicit local snapshot import** from operator
supplied source files. Import never runs inside the macOS LaunchAgent.

## Actual public directory specifications

Nasdaq Trader publishes `nasdaqlisted.txt` for Nasdaq securities and
`otherlisted.txt` for other listed securities, with symbol directory
metadata and a file creation footer. They change over time, including
intraday additions. Source definition:
https://www.nasdaqtrader.com/Trader.aspx?id=SymbolDirDefs

KRX Data Marketplace offers **전종목 기본정보** downloads:
https://data.krx.co.kr/
No undocumented KRX download endpoint, approval, license or private
credentials are assumed. The operator must review the KRX original
download and translate only **common stock** records to the explicit
`KRX_REVIEWED_EXPORT` four-column contract:

```csv
market,symbol,company_name,security_type
XKRX,005930,삼성전자,COMMON
```

The row above is a **format example**, not proof of current or historical
listing status, and its six-digit symbol and company name require review.

## Snapshot manifest and hashing

Use an independently reviewed local snapshot, saved next to a manifest
in a separate private folder (do not commit source snapshots, credentials
or hashes that could expose private provider contracts).

```json
{
  "schema_version": "listing_snapshot_v1",
  "sources": [
    {
      "kind": "NASDAQ_DIRECTORY",
      "filename": "nasdaqlisted.txt",
      "sha256": "<64 hex characters from shasum -a 256 nasdaqlisted.txt>",
      "observed_at": "<actual local first-observation ISO timestamp with UTC offset>",
      "source_id": "nasdaqtrader-direct-operator-reviewed"
    }
  ]
}
```

Each source requires a real **existing**, non-symlinked file, a SHA-256
matching the supplied raw bytes, a bounded source ID and a timezone-aware
`observed_at`. A hash proves file consistency *only*, not source
authenticity, licensing or that the file existed at an earlier date.
Never backdate `observed_at` to a company listing date.

Supported `kind` values and expected unmodified file columns:
- `NASDAQ_DIRECTORY`: pipe-delimited `nasdaqlisted.txt` with
  `Symbol|Security Name|Market Category|Test Issue|Financial Status|Round
  Lot Size|ETF|NextShares`, plus a `File Creation Time:` footer.
- `NYSE_DIRECTORY`: pipe-delimited `otherlisted.txt` with
  `ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test
  Issue|NASDAQ Symbol`, plus the footer. Only `Exchange=N` maps
  to XNYS; NYSE Arca/other venues are not silently reassigned.
- `KRX_REVIEWED_EXPORT`: operator-normalized CSV exactly
  `market,symbol,company_name,security_type`. Only XKRX,
  six-digit symbols and `security_type=COMMON` are accepted.

**Conservative US limitation:** the importer currently accepts only
non-test, non-ETF US records whose `Security Name` ends exactly
with ` - Common Stock`. This intentionally discards many real but
unreviewed ordinary shares, alternative share classes, ADRs, units,
warrants, debt, funds and differently formatted names, rather than
incorrectly treating them as normal operating companies. It is not
a complete US equity universe.

## Safe commands

Once the operator has downloaded and verified source files, preview
without writing anything:

```sh
market-event-listing-snapshot --manifest "/private/path/listings/manifest.json"
```

Generate an **exclusive-create** CSV only after reviewing the preview:

```sh
market-event-listing-snapshot \
  --manifest "/private/path/listings/manifest.json" \
  --write \
  --output "/private/path/listings/candidate-registry.csv"
```

The export feeds the earlier read-only `market-event-ticker-proposals`
CLI. A newly acquired snapshot has `valid_from=observed_at` and
`evidence_seen_at=observed_at`; both are **the first observed
snapshot time**, not a verified exchange listing inception. Therefore
old headlines received before that snapshot **cannot** be linked
retroactively. Aliases remain empty because name aliases are not
reconstructed by a live symbol directory.

A conversion step requiring only a user-attested observation time
cannot establish point-in-time listing history. Manual provenance
review, historical listing/symbol-change records, issuer type coverage,
licensing, independent first-observation evidence and market
symbol validation remain prerequisites before training or making
market predictions.

No network request, Ollama analysis, existing SQLite mutation,
LaunchAgent change, auto-approved link or price label occurs here.
