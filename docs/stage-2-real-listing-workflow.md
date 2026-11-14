# Actual listing source capture + Stage 2-C review bundle

Use one Mac shell command from the repository after the CI PR merges.
The `--update` option runs the existing full verifier (clean main fast-forward,
editable install, lint and tests) before this optional data workflow:

```bash
bash scripts/listing-research.sh --update --fetch-us
```

It explicitly downloads only two official Nasdaq Trader public HTTPS symbol
directories (`nasdaqlisted.txt` and `otherlisted.txt`), with no redirects,
a streaming 3 MB per-file limit, timeout and schema checks. A new private,
0700 directory under `~/Library/Application Support/TrisMarketEventIntelligence/listing-research/`
contains the unchanged source bytes, SHA-256 hashes, manifest, reviewed-required
issuer CSV, and report. If the existing default `news.sqlite3` exists, it
is only read to produce `proposals.json`. No news database is created or
modified. A new snapshot can only match events received **on or after**
the first actually observed source acquisition time. Past news usually
produces `FUTURE_REGISTRY_EVIDENCE` and zero candidates; **never backdate**
to force a match.

For KRX, independently obtain an actual official KRX Data Marketplace export
and manually normalize it to the established reviewed four-column contract.
Do not copy the illustrative example CSV from docs as production data.
The same one-command workflow then accepts the operator-normalized local file:

```bash
bash scripts/listing-research.sh --update --fetch-us \
  --krx-reviewed "/path/to/your/actual-normalized-krx.csv"
```

If an isolated review ledger already exists, pass `--review-ledger` alongside
`--news-db` to reconcile against exact candidate report bytes. Missing ledger
files are never created silently. Human approvals are neither fabricated nor
converted into historical price labels.

**Boundaries:** NasdaqTrader source names, expected symbols and publication
formats follow the public symbol directory description:
https://www.nasdaqtrader.com/Trader.aspx?id=SymbolDirDefs.
The source bytes can be hashed and parsed, but an HTTPS receipt is not
independent verification of issuer authenticity or listing inception.
Conservative ` - Common Stock` filtering excludes many legitimate names
and share classes. A file may exceed parsing bounds or change schema, in
which case the operation fails closed. KRX is **not** fetched by guessing an
undocumented authenticated API. This workflow does not import historical
listings, approve ticker relationships, validate reviewer identity, fetch
minute bars, train any model, alter LaunchAgent or start inference.

Output is create-only and stored outside the repository. Empty candidate
sets from old articles are expected, not errors. Review the `report.json`
source counts and missing-coverage limitations before claiming Stage 2 complete.
