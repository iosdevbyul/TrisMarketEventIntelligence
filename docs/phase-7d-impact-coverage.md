# Stage 4 / Phase 7-D — real-event coverage audit (read only)

This command answers **how many previously collected events are eligible**,
how many have versioned successful runs, how many are still pending, and
which safe error codes are currently recorded. It never calls Ollama,
requests another news feed, creates a database, modifies existing research,
or generates missing analysis. It does **not** claim semantic support,
factuality, correct direction, or prediction accuracy.

The report reuses the same \`is_eligible\` policy as \`ImpactProcessor\`.
For the selected model and prompt version, it distinguishes COMPLETED
from NO_HYPOTHESIS, and lists the pending-with-failure subset. Existing
v6 analyses remain visible in separate historical run counts; no v6 results
are recast as v7. \`last_failure_counts\` counts distinct currently recorded
event/error rows, **not** the number of failed inference attempts.
Sample IDs are deterministic and contain no article text.

## Intel Mac commands

\`\`\`sh
git switch main
git pull --ff-only origin main
python -m pip install -e ".[dev]"
python -m pytest -q tests/test_impact_coverage.py
python -m pytest -q
market-event-impact-coverage --model-id gemma2:latest
\`\`\`

To view v6 independently without re-executing inference:

\`\`\`sh
market-event-impact-coverage --model-id gemma2:latest \
  --prompt-version market-impact-v6
\`\`\`

Only after inspecting this report should a *separate explicit* manual
small-batch model run be considered. This feature has no implicit model,
scheduler, or DonghakStockVision integration. Historical SQLite v6 and v7
rows are never rewritten. A real quality assessment still needs independently
labelled multiple distinct source articles.
