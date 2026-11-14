# Stage 4 version transition comparison with one-command Mac checks

After event 13 produced a generic v7 impact and a v8 NO_HYPOTHESIS result,
this read-only audit compares **all eligible** events for the same model
across v7 and v8. It counts exact event-level transitions, missing attempts,
stored impacts, and conservative syntax warnings. No sources or model outputs
are exported, no inference is automatically invoked, and no SQLite mutation
takes place. A missing or partial database is reported as such, never created.

The statuses are descriptive. NO_HYPOTHESIS is not automatically more
accurate than COMPLETED; human evidence and causal review remains required.

## Mac workflow

From the project root use the same command every time:

```sh
bash scripts/verify.sh --update
```

The Bash entrypoint checks for a clean main and runs
`git pull --ff-only origin main` **before** loading the Python verification
module. It then runs install, Ruff and the complete test suite. On success
the JSON report includes `version_comparison` alongside other test results.
Output reports are under
`~/Library/Application Support/TrisMarketEventIntelligence/verification-reports/`.

If and only if you explicitly want live inference on a new versioned event:

```sh
bash scripts/verify.sh --update --live 7
```

This runs at most the explicitly selected 1–3 eligible IDs under the current
prompt. It cannot silently rerun already checkpointed successful IDs.
CI only tests mocked workflows; real local Ollama correctness requires
review of original sources and saved hypotheses.
