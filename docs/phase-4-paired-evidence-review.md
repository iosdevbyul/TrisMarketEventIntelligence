# Stage 4: paired human review for potential v8 false suppression

The first real v7/v8 comparison for economic event 13 changed from a
`COMPLETED` hypothesis to `NO_HYPOTHESIS`. That may be appropriate caution
or a false suppression. The pipeline cannot determine economic correctness
from the status change alone.

## One-command workflow

```sh
bash scripts/verify.sh --update
```

After install, Ruff and all tests, verification performs read-only v7/v8
status comparison as before. When matching, successfully checkpointed v7
and v8 results for the same model exist, it now also exports **up to five**
source-linked paired review cases to a new local JSON file in
`~/Library/Application Support/TrisMarketEventIntelligence/verification-reports/`.
The output path is printed in a `[verify] private paired review:` line.
Each record contains original local event evidence, source URL, both model
versions' stored rationales (or explicit empty hypotheses) and empty
`human_label` and `human_review_notes` fields. Changed outcomes are
prioritized and ordered by event ID. No Ollama call, SQLite write,
prediction or autonomous human judgment occurs; `quality_verified=false`.

The paired JSON contains **full source excerpts and model text** and is
created with owner-only filesystem permissions on macOS. Treat it as
private: only upload it for review deliberately. The ordinary summary
verification JSON stores only the local path and count, not source text.
Existing paired files are never overwritten, and past v6/v7/v8 data
are unchanged. If the local DB is missing or no paired checkpoint exists,
verification still finishes with a summary and does not generate a bundle.

Human reviewers should check the original reported claim, whether the v7
mechanism was justified, whether v8 abstention was warranted, and whether
missing evidence creates uncertainty. A NO_HYPOTHESIS output is not
automatically correct. Further release criteria require a representative
labeled cohort rather than one anecdotal example.
