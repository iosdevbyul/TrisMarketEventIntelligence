# Stage 1: model-analysis singleflight and freshness recheck

The real Mac operating report showed that article receipt to the persisted
Gemma checkpoint can be hours apart. Before promoting the code to an
automatically scheduled low-latency service, prevent accidental concurrent
model work and fresh-news inference from stale queue snapshots.

## Production safeguards

1. **Per-database singleflight**: `ImpactProcessor.process_pending` holds
   a nonblocking OS `flock` for the entire event selection, model call,
   verification and SQLite checkpoint sequence. Two shell/launchd/CLI
   invocations against the same database cannot process the same
   unchecked event concurrently. The lock path is adjacent to the DB:
   `news.sqlite3.analysis.lock`. No main branch mutation, service
   activation or code rearchitecture is needed.
2. **Contention is observable**: a competing process immediately
   returns `AnalysisStats.already_running=true`, with zero model
   calls, zero checkpoints and no false success. The direct analysis
   CLI exits with status 2 on contention. Existing default CLI JSON
   stays backwards-compatible when no contention occurs.
   The integrated collector→extraction→Gemma flow treats this
   condition as `blocked`, never `healthy`.
3. **Per-event freshness recheck**: in opt-in live analysis, every
   event's news receipt and publisher-claimed timestamp are checked
   using the actual current UTC instant *immediately before* its
   provider call. A slow first Ollama analysis cannot cause a
   second now-stale event to be sent based on the earlier batch
   start time. Expired events remain **deferred**, never checkpointed.
4. **Old research compatibility**: the live freshness gate stays
   opt-in. Historical analysis and explicit event validation
   behave as before; no currently stored model outputs are edited.

## Verified boundaries

Synthetic CI scenarios cover a held lock preventing model execution,
release followed by a single completed analysis, explicit event
selection while another process runs, time passing between two local
provider calls, and validation before any lock-file creation.
They run on Python 3.11, 3.12 and 3.13; they **do not** claim
verification of the user's private Mac/Ollama runtime, speed or
news completeness.

The existing `market-event-operational-evidence` records actual
per-call durations after a supervised local Gemma run. The user's
exercise time requires no Mac or Ollama interaction for this PR.
