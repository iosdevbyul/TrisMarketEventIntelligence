#!/usr/bin/env bash
# Read-only preview by default. --run explicitly invokes local Gemma4.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA="${HOME}/Library/Application Support/TrisMarketEventIntelligence"
AS_OF="${1:-$(date +%F)}"
MODE="${2:---preview}"
PERSIST="${3:-}"
DB="${MARKET_EVENT_DB:-$DATA/news.sqlite3}"
OUTPUT="${MARKET_PRESSURE_OUTPUT:-$DATA/market-pressure-first-real-v6.json}"

if [[ "$MODE" != "--preview" && "$MODE" != "--run" ]] || \
   [[ -n "$PERSIST" && "$PERSIST" != "--persist" ]] || \
   [[ "$MODE" != "--run" && -n "$PERSIST" ]]; then
  echo "Usage: bash scripts/verify-market-pressure.sh [YYYY-MM-DD] [--preview|--run] [--persist]" >&2
  exit 2
fi
if [[ ! -f "$DB" ]]; then
  echo "SOURCE_DB_MISSING: $DB" >&2
  exit 2
fi

args=(market-pressure-cycle --db "$DB" --as-of "$AS_OF"
      --lookback-days 10 --max-groups 2 "$MODE")
if [[ "$MODE" == "--run" ]]; then
  if [[ -e "$OUTPUT" || -L "$OUTPUT" ]]; then
    echo "OUTPUT_ALREADY_EXISTS: $OUTPUT" >&2
    exit 2
  fi
  args+=(--model gemma4:e2b --output "$OUTPUT")
  if [[ "$PERSIST" == "--persist" ]]; then
    args+=(--persist)
  fi
fi
bash scripts/news-learning-review.sh "${args[@]}"
if [[ "$MODE" == "--run" ]]; then
  jq '{
    status, modelCalls, inventory,
    events: [.events[] | {
      groupSourceIds,
      groupReason: .proposalExplanation,
      factsCount: (.analysis.groundedClaims | length),
      analysisStatus: .analysis.status,
      validationStage: .analysis.validationStage,
      validationCode: .analysis.validationCode,
      validationReasonCode: .analysis.validationReasonCode,
      invalidFields: .analysis.invalidFields,
      groundedClaims: .analysis.groundedClaims,
      noImpactAssessment: .analysis.noImpactAssessment,
      impacts: [.analysis.impacts[] | {
        market, sector, targetType, targetName, direction, strength,
        uncertainty, transmissionChannel, supportingSourceIds
      }]
    }]
  }' "$OUTPUT"
  echo "FULL_RESULT: $OUTPUT"
  echo "REQUIRES_HUMAN_SEMANTIC_REVIEW; NO_PRICE_FORECAST"
fi
