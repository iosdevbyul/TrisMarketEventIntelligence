#!/usr/bin/env bash
# Compare saved development-set scores. No model calls or overwrites.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DATA="${HOME}/Library/Application Support/TrisMarketEventIntelligence"
BASELINE="${1:-$DATA/historical-semantic-score-faa-v9-fda-v6-v2.json}"
CANDIDATE="${2:-$DATA/historical-semantic-score-faa-v10-fda-v7-v2.json}"
OUTPUT="${3:-$DATA/historical-semantic-regression-v9-v10-v1.json}"
if [[ -e "$OUTPUT" || -L "$OUTPUT" ]]; then
    echo "OUTPUT_ALREADY_EXISTS: $OUTPUT" >&2
    exit 1
fi
for file in "$BASELINE" "$CANDIDATE"; do
    if [[ ! -f "$file" ]]; then
        echo "SCORE_ARTIFACT_MISSING: $file" >&2
        exit 1
    fi
done
bash scripts/news-learning-review.sh historical-semantic-compare \
    --baseline "$BASELINE" --candidate "$CANDIDATE" \
    --output "$OUTPUT" | jq '{
        status,
        regressedMetrics,
        improvedMetrics,
        newlyFailingCases,
        newlyPassingCases,
        changedErrors,
        trustVerified
    }'
echo "COMPARISON_ARTIFACT: $OUTPUT"
echo "DEVELOPMENT_COMPARISON_ONLY; DOES_NOT_VERIFY_MODEL_TRUST"
