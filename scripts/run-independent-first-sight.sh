#!/usr/bin/env bash
# Phase 3 + 4: current official source capture, then one never-overwritten
# first-pass diagnostic of completely new event families.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DATA="${HOME}/Library/Application Support/TrisMarketEventIntelligence"
mkdir -p "$DATA"
PACKET="research/independent-first-pass-pilot-v1.json"
CAPTURE="$DATA/independent-first-sight-current-source-capture-v1.json"
RESULT="$DATA/independent-first-sight-gemma4-firstpass-v1.json"

if [[ -e "$RESULT" || -L "$RESULT" ]]; then
    echo "FIRST_PASS_ALREADY_EXISTS: $RESULT" >&2
    exit 1
fi

if [[ ! -e "$CAPTURE" ]]; then
    echo "Phase 3: verifying source excerpts on current official pages"
    bash scripts/news-learning-review.sh historical-first-sight \
        --packet "$PACKET" --capture-output "$CAPTURE"
fi

echo "Phase 4: running first-sight cutoff replay with installed Gemma4"
bash scripts/news-learning-review.sh historical-first-sight \
    --packet "$PACKET" --capture "$CAPTURE" \
    --model gemma4:e2b --output "$RESULT"

jq '{
    status,
    modelId,
    modelCalls,
    historicalPITVerified,
    generalizationVerified,
    observations: [
        .results[] | {
            eventId, asOfDateExclusive, questionId, primaryEvidence,
            decisionStage: .answer.decisionStage,
            implementationStatus: .answer.implementationStatus,
            publishedImplementationStatus,
            evidenceStatus: .answer.evidenceStatus,
            explanation: .answer.explanation,
            limitation: .answer.limitation
        }
    ]
}' "$RESULT"
echo "FIRST_PASS_ARTIFACT: $RESULT"
echo "CURRENT_PAGE_EXCERPTS_ONLY; HISTORICAL_PIT_NOT_VERIFIED; INDEPENDENT_REVIEW_REQUIRED"
