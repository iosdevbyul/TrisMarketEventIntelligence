# Whole-ledger ticker review integrity audit

PR #73 stores approved/rejected candidate reviews in a separate research-only SQLite file.
The listing command checks only rows returned by its filter and limit.
For an explicit read-only check of **all** rows (up to 100,000), run:

```bash
market-event-ticker-ledger-audit --ledger /path/to/private/reviews.sqlite3
```

The command checks SQLite integrity, ledger identity, every row's SHA-256, canonical
JSON, indexed identity columns, decision/status consistency, provenance fields and
explicit exclusion from point-in-time training. It never repairs, creates, or writes
to the ledger, production news database, LaunchAgent, or Ollama. Errors return
exit status 2; absence of output must not be treated as success.

A successful audit only proves internal record consistency, **not** the human
reviewer's identity, authenticity of supplied external evidence, correctness of
issuer mapping, historical availability of information, market impact or eligibility
for training. The underlying digest is not a tamper-proof digital signature;
a party able to rewrite both records and hashes can defeat the check. Safeguard
the file and provenance independently.

Do not run this command against `news.sqlite3`. Supply the separately initialized
review database created through `market-event-ticker-ledger --import-reviewed`.
