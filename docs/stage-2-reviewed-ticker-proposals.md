# Stage 2 — Point-in-time KR/US listed issuer candidates (review-only)

## Scope and trust boundary

This stage introduces an **offline, read-only proposal** route for connecting
a locally stored economic event to a potential Korean or US listed company.
This does **not** automatically tag existing news, insert ticker links, enable
Gemma, place trades, or activate prediction training.

The current always-on 15-minute macOS integrated RSS→event LaunchAgent is
unchanged. Existing `news.sqlite3`, event IDs, status files, and source
configurations are never written by this command.

**The operator supplies a reviewed company listing CSV.** The repository
does not invent issuer names, listing validity, historical symbols, or
their review time; no synthetic example below is imported in production.
The name match is a deterministic *candidate*, not proof of a genuine
ticker-to-event or event-to-price causal relationship.

## Listing registry contract

CSV columns (exactly):

```csv
market,symbol,canonical_name,aliases,valid_from,valid_until,evidence_seen_at,source_id
XNAS,MSFT,Microsoft Corporation,Microsoft,2026-01-01T00:00:00+00:00,,2026-01-01T00:00:00+00:00,EXAMPLE_NOT_VERIFIED
XKRX,005930,삼성전자,Samsung Electronics,2026-01-01T00:00:00+00:00,,2026-01-01T00:00:00+00:00,EXAMPLE_NOT_VERIFIED
```

**These rows and dates are illustrative only.** They are not a
verified point-in-time listing registry and must not be copied into
a real trading/research pipeline without authentic historical proof.

Fields:
- `market`: `XKRX`, `XNYS` or `XNAS`; exchange identity must be
  reviewed from a trusted listing authority
- `symbol`: exact uppercase ticker, retaining zero-padded Korean
  symbols and historical symbol intervals
- `canonical_name`: full reviewed issuer name
- `aliases`: zero or more explicit names separated by `|`; tickers
  themselves are **not matched** as aliases by default
- `valid_from`, `valid_until`: timezone-aware listing/name validity
  timestamps, half-open interval. Empty `valid_until` means ongoing
- `evidence_seen_at`: when the operator-provided identification
  evidence was actually available, not the time the CSV was generated
- `source_id`: nonempty human-reviewed source identifier or internal
  provenance key; **not** independently authenticated by this module

All timestamps must contain UTC offset. The parser rejects unknown
exchange codes, malformed identifiers, duplicate validity rows,
overlapping windows for the same exchange-symbol, oversized inputs,
bad alias definitions, and symlinked registries.

## Offline proposal

After sourcing an actual reviewed registry and saving it privately:

```sh
market-event-ticker-proposals \
  --db "$HOME/Library/Application Support/TrisMarketEventIntelligence/news.sqlite3" \
  --registry "/path/to/operator-reviewed-listings.csv" \
  --limit 100
```

The command reads **already extracted** economic events joined with
original article headlines from the existing SQLite database in
`mode=ro`; it does not issue RSS requests, initialize schemas,
modify DB rows, or write report files. Registry matches are
allowed only when:
1. company name appears in the **article headline** with appropriate
   Latin word boundaries or exact Korean text (summary-only matches
   are excluded);
2. historical listing interval includes local `detected_at`;
3. registry `evidence_seen_at` is no later than news `detected_at`;
4. one alias does not resolve to multiple market/symbol identities.
   Ambiguous names are excluded, not automatically disambiguated.

Result rows include `PROPOSED`, `AMBIGUOUS_ALIAS`,
`FUTURE_REGISTRY_EVIDENCE`, `FUTURE_NEWS_TIMESTAMP`,
or `NO_VERIFIED_NAME_MATCH`. Each proposed issuer is explicitly
`DIRECT_CANDIDATE` with `human_review_required=true`.
The entire batch is `REVIEW_REQUIRED`; there are **no approved
EventTickerLink records, automatic labelling, backfills, or training
dataset eligibility** from the candidate matcher.

### What remains before real Stage 2 completion

- Import and audit a genuine licensed, versioned KR/US listed issuer
  registry, including corporate name changes and issuer vs ETF
  classification. Historical `evidence_seen_at` cannot be
  reconstructed reliably from a freshly downloaded current list.
- Human review of false positives (including common English corporate
  names), false negatives, issuer subsidiaries, cross-market duplicate
  listings, and indirect macro/sector exposure.
- Persist only an operator-confirmed, provenance-backed link with a
  reviewed point-in-time timestamp (separate later change).
- Verify linked symbols, sessions, halts, and *adjusted minute-bar*
  provenance from licensed sources before learning reactions.
- Evaluate event news coverage and prediction calibration out of
  sample; no return, direction or trading recommendation is produced.

The older offline event-ticker-screen already consumes **explicit
user-supplied** EventTickerLink rows for research. The new proposal
output is NOT that approved schema and cannot silently become it.
