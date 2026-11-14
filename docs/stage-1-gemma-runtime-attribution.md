# Stage 1: actual local Gemma queue versus inference time

## Goal

Resolve the actual Mac evidence: the old 13 persisted impact checkpoints
were **hours** after news receipt, but these database timestamps do not
show how long Ollama generated text versus how long events waited for
an operator or periodic task to run. Do not claim a model took hours
to respond based on this end-to-end latency alone.

## New instrumentation and audit

`ImpactProcessor` now appends one `impact_attempt_timings` row per
actual model attempt. This is a new append-only table; no existing
`impact_runs` checkpoints or `impact_errors` are rewritten.
It records:

- `attempted_at`: UTC when the event actually entered the provider path
- `provider_returned_at`: UTC after the provider call finished *or failed*
- `completed_at`: UTC after validation, persistence or error recording
- `provider_elapsed_ms`: monotonic-clock duration including Ollama HTTP,
  inference and response parsing, also for failures/timeouts
- `processing_elapsed_ms`: monotonic total invocation plus validation
  and checkpoint/error persistence wall time
- `outcome`: COMPLETED, NO_HYPOTHESIS, ERROR

The privacy boundary is strict: never append prompts, responses,
article text, credentials, full exception messages or network URLs.
Freshness-deferred and ineligible events **do not invoke the model**
and do not add false model attempts.

The offline read-only `audit_gemma_timing` joins this table to the
existing event extraction timeline. By source, it reports count,
median and 95th percentile for:

1. Economic event extraction → actual model attempt: queue/operator delay
2. Actual provider call: Ollama/LLM wall time
3. Provider return → validation + local result checkpoint
4. Overall attempted analysis duration

The audit reports legacy DBs without the new table as
`LEGACY_TIMING_UNAVAILABLE`. **Do not infer old Gemma calculation time**
from `impact_runs.processed_at`. Model-attempt failures remain
counted and timed, not silently removed. No first-token measurement
or independently verified realtime detection claim.

This audit is integrated into the existing
`market-event-operational-evidence` report. The stage-1 evidence
review now explicitly lists `GEMMA_PROVIDER_DURATION_NOT_YET_OBSERVED`
until a future actual model call is measured.

## Operator next step

No need to run any commands for code validation: GitHub CI verifies
mock providers, normal/failed model calls, old DB compatibility,
read-only audit, anomalous chronology and combined operational report
on Python 3.11, 3.12 and 3.13.

**Actual model performance still requires a separately authorized
local execution** on the Mac. The pre-existing
`bash scripts/verify.sh --live-next 1` can request a bounded new
eligible model run under operator control, but it installs development
dependencies and invokes Ollama; do not run it as part of the offline
audit or use it to manufacture a market-session result.

After a real, supervised local model invocation, the existing
`bash scripts/audit-operations.sh` exposes newly measured provider
durations without invoking the model. This measurement alone cannot
verify trading-time news completeness or stock-price prediction.
