# Stage 4 — expand real v8 Gemma 2 validation to fresh economic events

## Single Mac command (explicit real inference)

```sh
bash scripts/verify.sh --update --live-next 3
```

This opt-in command first fast-forwards a clean main, installs dependencies,
runs Ruff and **all** Python tests. It then queries the existing local SQLite
database **read-only** for at most three eligible `REPORTED` economic events
that have **never been attempted by this model under any prompt version**.
It favors different economic event classifications and excludes existing
completion checkpoints, prior failures, rumors, and ineligible policy reports.
The actual candidate IDs are selected from the local database, never invented.
No unattended recurrent job is created and inference is not invoked by
`bash scripts/verify.sh --update` alone.

Once selected, Ollama `gemma2:latest` is checked and the normal bounded
`--event-ids` pipeline analyzes exactly those events, retaining model+v8
versioned checkpoint safeguards. The run emits saved outcome counts and
v7 comparison reports. Failed events remain failures, not false successes.

Because new events may have **no v7 output**, the workflow also generates a
private `cohort-review-*.json` containing each selected event's original
news excerpt, source URL, v8 model rationale (or explicit NO_HYPOTHESIS),
and empty human review fields. This differs from paired v7/v8 review
(which requires both versions for a given event). The file is owner-only
(0600 on macOS), never overwritten, and its contents are **not** copied
into the ordinary verify summary JSON or echoed to terminal.

Report files are placed in
`~/Library/Application Support/TrisMarketEventIntelligence/verification-reports/`.
The summary prints `[verify] private new cohort review:` with its exact path.

After the first successful batch, running the same command again advances
to other never-before-attempted eligible events; if none exist, it does
not call Ollama or fabricate new data. You may use `--live-next 1` or
`--live-next 2` for smaller explicit batches. The older `--live 13` remains
available for a specific versioned event, but cannot be combined with
`--live-next`.

CI tests this using mocked model calls, never real Gemma 2. An accepted
hypothesis is not a price prediction, and NO_HYPOTHESIS is not automatically
better; only source-linked independent review can establish semantic quality.
