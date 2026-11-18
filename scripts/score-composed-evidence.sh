#!/usr/bin/env bash
# Evaluate existing local FAA/FDA Gemma4 output without another Ollama call.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DATA="${HOME}/Library/Application Support/TrisMarketEventIntelligence"
FAA="${1:-$DATA/faa-evidence-composed-v7.json}"
FDA="${2:-$DATA/fda-evidence-composed-v4.json}"
OUTPUT="${3:-$DATA/historical-semantic-score-faa-v7-fda-v4-v1.json}"
if [[ -e "$OUTPUT" || -L "$OUTPUT" ]]; then
    echo "OUTPUT_ALREADY_EXISTS: $OUTPUT" >&2
    exit 1
fi
for source in "$FAA" "$FDA"; do
    if [[ ! -f "$source" ]]; then
        echo "SAVED_MODEL_OUTPUT_MISSING: $source" >&2
        exit 1
    fi
done
bash scripts/news-learning-review.sh historical-semantic-score \
    --benchmark research/historical-semantic-benchmark-v1.json \
    --faa "$FAA" \
    --fda "$FDA" \
    --output "$OUTPUT"
echo "SCORE_ARTIFACT: $OUTPUT"
echo "NO_MODEL_CALLS; RESULT_REQUIRES_INDEPENDENT_HUMAN_REFERENCE_REVIEW"
