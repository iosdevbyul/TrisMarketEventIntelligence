# Stage 2-C: Validate captured real listed issuers against actual news

The owner's Mac successfully captured **3,736** review-required US names
(2,231 XNAS + 1,505 XNYS) from live NasdaqTrader directories at
`2026-11-14T15:54:40.590632+00:00` and inspected 70 events, with 0
candidates. That observation is not independent historic listing proof.

After merging this PR, use the existing verified Mac workflow:

```bash
bash scripts/verify.sh --update
bash scripts/validate-listings.sh
```

Or on subsequent runs, use
`bash scripts/validate-listings.sh --update` as a single command.
The script finds the newest real `observed-*` source bundle in the
private Application Support directory, **does not redownload**, verifies
each raw file against SHA-256 recorded in `manifest.json`, and confirms
the review registry is exactly reproduced from those bytes. No stale,
altered or unrelated registry may be substituted unnoticed.

The existing live news SQLite is queried with `mode=ro` and
`PRAGMA query_only`; 500 events per batch in *ascending* ID order,
up to 16 pages (8,000 events) per run. Pagination exposes
`has_more_events` and `resume_after_event_id` on a partial report
and makes no completeness claim past the bounded scan. All
`proposals-page-NNN.json` files remain immutable, independent
human-review bundles with individual SHA-256 fingerprints.

`summary.json` explains whether 0 candidates arose among old news
(before first source observation) or news after observation, with status
counts: `FUTURE_REGISTRY_EVIDENCE`, `FUTURE_NEWS_TIMESTAMP`,
`NO_VERIFIED_NAME_MATCH`, `AMBIGUOUS_ALIAS`, `PROPOSED`.
A post-observation exact issuer mention can produce a
`DIRECT_CANDIDATE` without approving it. `--review-ledger` is
optional; where present it audits the entire separate ledger and
reconciles review decisions bound to the **exact page digest**.
Without a supplied ledger, all real candidates remain pending.

A historical listing cannot be inferred from a current directory;
past articles must **not** be force-matched by backdating
`valid_from` or `evidence_seen_at`. The strict full-name policy
under-matches abbreviated headlines; this is measured as a source
coverage problem, **not** worked around with speculative aliases.
No automatic trading signals, causal labels, model calls, changes to
the 15-minute collector, or writes to live news SQLite occur.

If KRX source files are later supplied with actual independent evidence,
they can also be captured using the existing
`scripts/listing-research.sh --krx-reviewed <file>` path. This stage
does not invent a KRX dataset.


## Stage 2-C name-coverage diagnosis and fresh-news mode

The first operator execution (2026-11-14 UTC) inspected 70 events with
`NO_VERIFIED_NAME_MATCH=70`, all received **before** the US listing
snapshot. This alone does not mean the news collector is broken and does
not justify backdating the 3,736 company listing records.

With this stage, run the **same existing shell script** after merging:

```bash
bash scripts/validate-listings.sh --update
```

The existing full scan remains the default. Its private, immutable
`alias-diagnostics-page-NNN.json` files report deterministic
**possible legal-suffix alias mentions** only when the original
strict name matcher did not match. For instance, a reviewed source
may contain `Microsoft Corporation` while a headline says
`Microsoft`. The alias hint is a review item, **never** a new
`DIRECT_CANDIDATE`, an accepted alias, or an approved relationship.
We deliberately do not strip terms like `Holdings` or `Group`, do
not use ticker text as evidence, and skip very short Latin names.
Aliasing remains incomplete and may create false positives, including
multiple issuers with the same shortened form.

The summary now reports `diagnostic_alias_mentions`,
`diagnostic_alias_status_counts`, a bounded set of examples, and
`diagnostic_aliases_approved=0`. A hint has explicit review and
point-in-time safety flags. Headline examples stay exclusively in the
private output folder, never uploaded to GitHub.

To inspect **only news first received after the source snapshot**:

```bash
bash scripts/validate-listings.sh --post-snapshot-only
```

This uses SQLite `julianday()` timestamp comparisons (rather than
lexical string comparison across timezones), retaining the original
append-only IDs and exact evidence windows. If no such event has
arrived, it reports `events_inspected=0` and
`new_news_available=false` without creating synthetic articles or
starting the collector. Subsequent executions can inspect new events
without fetching the listings again.

**Review boundary:** `proposals-page-*.json` remains the only
existing format supported by the exact-hash human review ledger.
`alias-diagnostics-page-*.json` is deliberately a different schema:
the existing approval CLI must not treat any hint as an approved
EventTickerLink. Independent identity/source verification and
operator-reviewed aliases are future work; no automatic mutation of
the registry or live news SQLite is allowed.
