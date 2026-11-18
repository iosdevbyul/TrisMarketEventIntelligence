#!/usr/bin/env bash
# Score FAA v8 and FDA v5 against the explicit split-axis development reference.
# No Ollama inference; do not overwrite the earlier v7/v4 score.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DATA="${HOME}/Library/Application Support/TrisMarketEventIntelligence"
FAA="${1:-$DATA/faa-evidence-composed-v8.json}"
FDA="${2:-$DATA/fda-evidence-composed-v5.json}"
OUTPUT="${3:-$DATA/historical-semantic-score-faa-v8-fda-v5-v2.json}"
if [[ -e "$OUTPUT" || -L "$OUTPUT" ]]; then
    echo "OUTPUT_ALREADY_EXISTS: $OUTPUT" >&2
    exit 1
fi
for file in "$FAA" "$FDA"; do
    if [[ ! -f "$file" ]]; then
        echo "SAVED_MODEL_OUTPUT_MISSING: $file" >&2
        exit 1
    fi
done
bash scripts/news-learning-review.sh historical-semantic-score \
    --benchmark research/historical-semantic-benchmark-v2.json \
    --faa "$FAA" --fda "$FDA" --output "$OUTPUT"
echo "SCORE_ARTIFACT: $OUTPUT"
echo "NO_MODEL_CALLS; DEVELOPMENT_ONLY; INDEPENDENT_HOLDOUT_REQUIRED"
