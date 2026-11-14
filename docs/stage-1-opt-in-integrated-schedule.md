# Stage 1 — safe opt-in integrated macOS polling and watchdog

## Why this exists

On the owner's Intel Mac, eight RSS feeds completed a manual bounded
collector cycle in approximately 2.45 seconds and persisted 467 articles
and 38 economic events. The existing 13 Gemma checkpoints lagged behind
receipt by hours, but model runtime and scheduler wait cannot be
distinguished for historical checkpoint-only rows. The previous
RSS-only collector LaunchAgent runs the **collection cycle**, not the
integrated collection→extraction→optional model workflow.

This is a **candidate operational schedule**, not an installed or
activated service. No background inference is enabled by merging it.

## Explicit replacement, not a second collector

The integrated config generator
`market-event-integrated-launchd` produces a macOS user LaunchAgent
running the previously tested `market-event-integrated` entrypoint.
It uses the same SQLite DB as the old collector, but a **distinct**
status file (`integrated-last-run.json`) and logs. Generation
does not contact a feed, poll the database, invoke Ollama, call
`launchctl`, install an agent or alter an existing service.

- Default model calls: **off**. Integrated collection and economic
  event extraction are enabled only *after explicit operator activation*.
- Default proposed cadence: **900 seconds** (15 minutes), configurable
  from **300 to 86400 seconds**. Faster periods produce more HTTP requests
  and cannot establish actual publisher-first-release timeliness.
- `RunAtLoad = false` and `--write` only **create a new plist**
  with mode 0600. No auto-bootstrap, service enable, login-item
  change, background model provisioning or overwrite.
- An existing legacy RSS-only collector plist
  `com.tris.market-event-intelligence.plist` prevents write.
  **The operator must first inspect, stop and remove the old
  collector deliberately**; the new generator does not change it.
  Presence of an absent plist is not proof a job is unloaded:
  inspect `launchctl` separately before any activation.
- Enabling the model requires *both*
  `--enable-local-analysis` and
  `--confirm-background-inference` plus an installed local
  model name through `--model`. The only serialized environment
  value is `MARKET_EVENT_LLM_MODEL`; never put API secrets in a plist.
  Model inference still uses loopback Ollama and the prior hard
  bounds (up to three new eligible events per pass).
- The prior cross-process model lock, per-event live freshness
  recheck, failed provider recording and bounded extraction remain
  effective. No missed-run replay or unbounded historical AI queue
  is introduced.

## Commands are reference documentation, **not an instruction to run now**

Preview plist XML with **no writes**:

```sh
market-event-integrated-launchd --interval 900
```

Create only a new config, **after separately confirming the old
collector is disabled and its plist no longer exists**:

```sh
market-event-integrated-launchd --interval 900 --write
```

An optional model-enabled preview requires all three deliberate
options (do not schedule until supervised Mac latency tests pass):

```sh
market-event-integrated-launchd --enable-local-analysis \
  --confirm-background-inference --model gemma2:latest
```

**No commands above automatically activate the agent.** A future
operator-reviewed activation would require inspecting existing
`launchctl` state, ensuring only one collector job runs against the
DB, creating the log directory, and explicitly bootstrapping a newly
reviewed plist. The user is currently away from the Mac, so no
activation, replacement or model invocation is attempted.

## Read-only schedule watchdog

`market-event-integrated-watchdog` checks the last integrated
checkpoint in the existing local private application support folder.

```sh
market-event-integrated-watchdog --max-age-minutes 30
```

The watchdog compares the recorded, timezone-aware completion time
with the audit time. It reports `OBSERVED_RECENT_SUCCESS` only if
the status is a recent integrated run, the collection phase and
**every expected source** succeeded (HTTP 200 or 304), and event
extraction was recorded. Missing, corrupt, stale, disabled or partially
successful runs produce `BLOCKED` and a stable reason; exit code 2
is intentional. A zero-new-article successful poll is not a failure.

This is a **local, read-only** checkpoint test: it neither checks
external headline completeness nor actually watches the machine in
the background. It never starts a service, opens Ollama, requests
RSS URLs or modifies the local DB.

## Verification and real operating gates

CI mock tests validate:
- no background model default and no implicit launch activation;
- 5-minute minimum interval, correct sources, absolute paths;
- separate deliberate inference consent and local-only model name;
- refusal to overwrite a pre-existing plist or replace the legacy agent;
- a true mocked integrated RSS pass producing healthy status and
  a readable watchdog observation;
- missing, corrupt, stale, future-dated, wrong-source or degraded
  status detecting a blocker, with no network requests.

The full user goal remains KR/US live news detection, point-in-time
ticker linkage, genuine exchange-specific 1/5/15/30/60-minute price
reaction learning and only then forward prediction. No stage can
be considered complete on generated plists or GitHub CI alone.
Mac sleep, LaunchAgent run frequency, rate limits, supervised
Ollama speed, independently confirmed news timeliness, ticker
accuracy and real licensed minute data remain unverified.
