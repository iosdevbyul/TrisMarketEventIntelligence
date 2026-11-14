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


## Stage 2-D: real duplicate causes and collector-to-event observability

The Mac source scan found that 7 alias hints corresponded primarily to
repeated BBC Microsoft outage headlines under distinct event IDs. Repeated
event IDs **are not evidence** of seven independent market impacts.

This release makes **future article ingestion** idempotent for the exact
`(source, url)` pair even when the RSS GUID changes, while preserving the
existing `(source, external_id)` uniqueness and raw capture evidence.
Different publisher sources and different URLs are never guessed identical.
The historical article/event records are **not deleted, migrated, renumbered,
deduplicated, or backfilled**.

Use the same installed Mac script:

```bash
bash scripts/validate-listings.sh --update
```

The private `summary.json` now includes `news_pipeline_evidence`:
article/event row totals, article first-seen count after the captured US
symbol snapshot, latest recorded RSS fetch attempts by source, exact same
source/URL repeated articles, grouped canonical URL story candidates,
and distinct economic events attached to the same article.
The report reads existing SQLite with `mode=ro` and
`PRAGMA query_only=ON`; no collector invocation or data mutation occurs.

This distinguishes a successful polling attempt from a newly stored
article, and a newly stored article from an event generated by an
extraction pathway. The scheduled RSS operational cycle handles claim
extraction and story grouping; it does not automatically prove that
`economic_events` has been materialized for every new article. That is
an integration-readiness gap, not a reason to fabricate events or
silently invoke a different event processor on the operator's live DB.

Canonical URL groups are **diagnostics only**: removing tracking params
is not proof of article equivalence across websites or versions.
Any older duplicate articles and associated event IDs remain immutable.
A current successful polling checkpoint does not prove complete news
coverage, the 15-minute scheduler is active, or historical market impact.
The real data counts must be checked on the owner's Mac after deployment.


## Stage 2-E: opt-in post-snapshot event materialization

The prior pipeline audit identified a real integration distinction: the
ordinary scheduled RSS collection cycle records claims and story groups,
but the separate integrated pipeline performs economic event extraction.
A successful collection cycle is **not** proof of new economic event IDs.

The existing Mac validation script now supports **explicit** bounded
materialization of previously stored articles received after the issuer
snapshot:

```bash
bash scripts/validate-listings.sh --update --materialize-new-events
```

This deliberately **writes** to the existing local `news.sqlite3`:
it may append `economic_events`, `event_metadata` and
`extraction_runs` records for at most **500 previously unprocessed
post-snapshot articles** per invocation. This is not the original
read-only validation mode. It retains all old articles, old event IDs
and review state. It never replays old news by backdating listing
evidence, and it does not approve issuer relationships.

The execution obtains the same `.pipeline.lock` used by the
integrated collector, verifies the actual listing manifest and
source-file digests, restricts article `detected_at` to the observation
window, performs existing rule-based event extraction only, and then
runs the existing **read-only post-snapshot ticker review validation**.
Its JSON result distinguishes articles processed, events created,
and name-matched review candidates. Repeating the command is
checkpoint-idempotent; hitting the 500-article cap explicitly signals
that more backlog may exist.

The mode does **not** make RSS network requests, invoke Gemma/Ollama,
start launchd, alter the collector schedule or claim historical
listing coverage. It is intentionally **manual**, not another hidden
scheduler. The separate existing command remains fully read-only:

```bash
bash scripts/validate-listings.sh --post-snapshot-only
```

Operators should inspect the explicit `event_extraction` counts and
`validation_summary` in the private output folder rather than
interpreting a successful fetch as successful event materialization.


## Stage 2-F: one operator-invoked integrated RSS → ticker review pass

Instead of creating another competing collector or modifying the existing
15-minute LaunchAgent, Stage 2-F reuses the existing `run_integrated_pass`
implementation (which already collects RSS and materializes economic events)
and links **previously captured, digest-verified** listed issuers to its
newly extracted news. The new mode is deliberately **not** installed in
launchd and never enables model inference.

After this PR is merged, the owner's existing script can pull, test,
collect, extract, and generate private ticker-review artifacts in one run:

```bash
bash scripts/validate-listings.sh --update --integrated-new-news
```

This command performs actual RSS network requests and can append **new**
articles, extraction checkpoints, and economic events to the existing
`news.sqlite3`. It is **not** the normal read-only audit. The opt-in
switch is required; the legacy script and LaunchAgent defaults are
unchanged. It requires an already-captured `observed-*` listing bundle,
verifies the source SHA-256 and registry before requesting RSS, uses the
same integration lock to avoid concurrent executions, and restricts
event extraction to articles first received after that listing capture.
There is no historical backdating or automatic human ticker approval.

The integrated result adds `ticker_review` fields: exact report
`summary_path`, `events_inspected`, `candidates`, `alias_hints`,
`has_more_events`, and `automatic_approvals=0`. Every run emits a
new create-only `integrated-*` private folder inside
`~/Library/Application Support/TrisMarketEventIntelligence/listing-research/`.
An empty candidate set is a valid result, especially when publishers
supply no new issuer-related articles. A candidate still requires
independent human review and cannot be used as a verified training label.

The CLI refuses model inference alongside issuer review; it never calls
Ollama for this mode. Existing `--enable-local-analysis` behavior
outside the new option is preserved and still requires explicit
activation. No services, secrets, market-minute feeds or scheduled
tasks are created. The integrated RSS stage should not be invoked
concurrently with an existing scheduler without first checking its
operational status; the lock prevents two integrated passes, while
the ordinary collector retains its separate lock.

Read-only alternatives remain:
`bash scripts/validate-listings.sh --update` for all existing events
and `bash scripts/validate-listings.sh --post-snapshot-only` for newer
events already stored. Unlike those modes, `--integrated-new-news`
can write append-only data.

**Evidence gap:** This release is CI-tested with mock RSS and source
snapshots. Real live publisher results, actual ticker candidates,
ongoing scheduler state and market coverage must be checked using the
owner's Mac logs before declaring production readiness.
