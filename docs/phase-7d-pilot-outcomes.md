# Stage 4 / Phase 7-D — per-event outcome comparison

After explicitly running 1–3 real-news event IDs, compare each selected
event's **stored** model outcome without re-running Gemma 2:

- COMPLETED: versioned checkpoint exists; not a guarantee of analysis accuracy.
- NO_HYPOTHESIS: completed without a market-impact hypothesis.
- FAILED_PENDING_RETRY: saved sanitized error code, no success checkpoint.
- NOT_ATTEMPTED: no result or failure for that exact model/prompt version.
- EVENT_NOT_FOUND: ID is absent from the selected local SQLite database.

Output also includes stored impact counts, direction and uncertainty
distributions and conservative rationale syntax warnings. Do not treat a
legacy v6 record as a v7 success. Never expose article bodies or raw
provider error messages. This report does not label actual economic truth,
causal validity, or future market movements.

## Intel Mac instructions

\`\`\`sh
git switch main
git pull --ff-only origin main
python -m pip install -e ".[dev]"
python -m pytest -q tests/test_impact_pilot_report.py
python -m pytest -q
export MARKET_EVENT_LLM_MODEL=gemma2:latest
market-event-analyze --pilot-preview
\`\`\`

If you have already manually run some selected candidate event IDs, inspect
those exact IDs without starting Ollama:

\`\`\`sh
market-event-pilot-report --event-ids 6,7 \
    --model-id gemma2:latest
\`\`\`

The \`6,7\` IDs above are examples; use IDs from **your own** selected
pilot. Compare historical stored v6 separately using:

\`\`\`sh
market-event-pilot-report --event-ids 6 \
    --model-id gemma2:latest --prompt-version market-impact-v6
\`\`\`

To actually run a new bounded sample, first preview candidates and use
\`market-event-analyze --provider local --event-ids ... --batch-size ...\`
as documented in \`phase-7d-bounded-pilot.md\`. This read-only report does
not initiate an inference, change any SQLite record or certify model quality.
