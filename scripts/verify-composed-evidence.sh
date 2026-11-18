#!/usr/bin/env bash
# Run both independent, prospective historical source-composition checks.
# Never overwrite the saved FAA v1-v9 or FDA v1-v6 comparison outputs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA="${HOME}/Library/Application Support/TrisMarketEventIntelligence"
mkdir -p "$DATA"

run_case() {
    local name="$1" chain="$2" tracks="$3" early="$4" late="$5" output="$6"
    if [[ -e "$output" || -L "$output" ]]; then
        echo "OUTPUT_ALREADY_EXISTS: $output" >&2
        return 1
    fi
    echo "Starting prospective $name cutoff-scoped Gemma4 replay"
    bash scripts/news-learning-review.sh historical-event-chain \
        --chain "$chain" \
        --question-tracks "$tracks" \
        --sequence "$early" "$late" \
        --reasoning-mode staged-composed \
        --run \
        --model gemma4:e2b \
        --output "$output"

    jq '{
        status,
        modelCalls,
        historicalPITVerified,
        questions: [
            .replaySteps[] as $step |
            $step.assessment.independentQuestionAxes.questionAssessments[] | {
                cutoff: $step.asOfDateExclusive,
                questionId,
                primary: {sourceId, claimId},
                baselinePrimaryEvidence,
                regulatoryDecisionStage,
                requestedEvidenceStatus,
                directRecordAvailability,
                indirectRegulatoryProgress,
                implementationCompletionStatus,
                modelReportedImplementationCompletionStatus,
                evidenceStateConsistencyFlags
            }
        ]
    }' "$output"
}

run_case "FAA" \
    "research/faa-7379max-january-2024-event-chain.json" \
    "research/faa-2024-question-lifecycle.json" \
    "2024-01-13" "2024-01-25" \
    "$DATA/faa-evidence-composed-v10.json"

run_case "FDA" \
    "research/fda-covid-vaccine-2023-event-chain.json" \
    "research/fda-2023-question-lifecycle.json" \
    "2023-06-17" "2023-10-04" \
    "$DATA/fda-evidence-composed-v7.json"

SCORE="$DATA/historical-semantic-score-faa-v10-fda-v7-v2.json"
if [[ -e "$SCORE" || -L "$SCORE" ]]; then
    echo "OUTPUT_ALREADY_EXISTS: $SCORE" >&2
    exit 1
fi
bash scripts/news-learning-review.sh historical-semantic-score \
    --benchmark research/historical-semantic-benchmark-v2.json \
    --faa "$DATA/faa-evidence-composed-v10.json" \
    --fda "$DATA/fda-evidence-composed-v7.json" \
    --output "$SCORE" >/dev/null
jq '{
    status, rates, trustGate,
    failedCases: [
        .caseResults[] | select(.status != "PASS") |
        {id, errors, primaryEvidenceObserved, decisionObserved,
         evidenceObserved, modelReportedCompletion}
    ]
}' "$SCORE"
echo "FULL_SCORE_ARTIFACT: $SCORE"
echo "BOTH_CASES_GENERATED_REQUIRES_HUMAN_SEMANTIC_REVIEW"
