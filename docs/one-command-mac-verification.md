# One-command Mac verification (stage 6; validates stage 4)

Run once from the project directory, after pulling this PR's merged main:

```sh
bash scripts/verify.sh --update --live 13
```

This command requires clean `main` for `--update`. It performs a fast-forward-only
Git pull, editable dev install, Ruff, the **complete** pytest suite, local
Ollama readiness check, explicit v8 analysis of event 13, and a version-separated
read-only comparison with v7. Existing versioned v6/v7 analyses are never
overwritten. An existing v8 checkpoint for event 13 is rejected before
inference by the existing pilot guard; use a new eligible event ID instead.

Offline-only verification (no Ollama invocation or SQLite analysis):

```sh
bash scripts/verify.sh --update
```

To analyze other explicitly selected events, use `--live 7,13` with at most
three unique positive IDs. Never run the entire news database implicitly.

Every completed workflow saves a JSON report to:

`~/Library/Application Support/TrisMarketEventIntelligence/verification-reports/`

The JSON report includes test status, model/version, execution statistics
and per-event outcome summaries; no source article bodies or API keys.
Syntax checks do not prove model economic validity or investment accuracy.

Unsafe Git states cause refusal rather than reset, checkout, stash, or force.
CI validates the command offline with mocked Git and Ollama; CI does not
run the model. For first-time script availability, run `git pull --ff-only`
on main before invoking `bash scripts/verify.sh`.

## tmux or a new terminal shows `(base)`

The launcher detects the current interpreter's project installation. If the
current `python` cannot import the verification module, it tries the local
`.venv/bin/python`, then the existing `tris-market-event` Conda environment
via `conda run`. It never installs into or mutates Conda `(base)`.
This keeps the ordinary `bash scripts/verify.sh --update` command valid in
tmux without manually activating an environment. If none of those environments
has the package, the launcher exits with a clear activation hint.

A shell already running a pre-fix version must fetch the updated launcher once
before running it: `git pull --ff-only origin main && bash scripts/verify.sh --update`.
