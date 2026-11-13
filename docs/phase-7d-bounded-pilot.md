# Stage 4 / Phase 7-D — bounded real-news Ollama pilot

Previous batch analysis orders events by ascending ID; after a v7 failure
event 6 may be selected again. This workflow first previews **distinct
eligible, unattempted real economic events** from the user's existing SQLite
without modifying data or invoking Ollama. It favors different categories
and excludes prior failed events from the automatic preview. It does not
verify the news source or guarantee market accuracy.

## Mac verification

\`\`\`sh
git switch main
git pull --ff-only origin main
python -m pip install -e ".[dev]"
python -m pytest -q tests/test_impact_pilot.py
python -m pytest -q
export MARKET_EVENT_LLM_MODEL=gemma2:latest
market-event-analyze --pilot-preview
\`\`\`

If the preview lists candidates, choose at most three **actual listed IDs**.
Explicit execution is separate and requires the installed local model:

\`\`\`sh
market-event-analyze --provider local --event-ids ID1,ID2 \
  --batch-size 2 --timeout-seconds 300
market-event-analyze --show-errors
market-event-impact-coverage --model-id gemma2:latest
\`\`\`

Replace \`ID1,ID2\` with returned integer IDs, not literal text. Existing
checkpoints and ineligible or unknown IDs are rejected before invoking any
model. For selected IDs the run is bounded to three; a failed result is
recorded as an error, never a completed hypothesis. Preserve historical
v6/v7 runs and the previous manual review CSV. A subsequent independent
review export should use a new file path; existing CSVs are protected from
overwrite. CI mocks model outputs: only the Intel Mac can evaluate real
Gemma 2 analytical quality.
