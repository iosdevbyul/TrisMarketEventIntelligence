#!/usr/bin/env bash
# Inspect private Gemma4 stage diagnostics. Do not upload the raw trace.
set -euo pipefail
TRACE="${1:?Usage: bash scripts/inspect-market-pressure-trace.sh PRIVATE_TRACE_JSONL [--raw]}"
MODE="${2:---summary}"
if [[ ! -f "$TRACE" ]]; then
  echo "TRACE_NOT_FOUND: $TRACE" >&2
  exit 2
fi
if [[ "$MODE" == "--raw" ]]; then
  jq -s '[
    .[] | select(
      .event == "LLM_REQUEST" or
      .event == "LLM_RESPONSE" or
      .event == "VALIDATION_REJECTED" or
      .event == "LLM_EXCEPTION"
    ) | {
      sequence, event, stage, call, eventId, modelId,
      systemPrompt, userInput, requestedSchema,
      rawResponse, fieldErrors, errorMessage, traceback
    }
  ]' "$TRACE"
else
  jq -s '[
    .[] | {
      sequence, event, stage, call, eventId, modelId,
      promptSha256, responseSha256, responseChars,
      claimCount, acceptedImpactCount, verifiedImpactCount,
      errorType, errorMessage, fieldErrors,
      validationCode, validationReasonCode,
      analysisStatus, status, modelCalls
    } | with_entries(select(.value != null))
  ]' "$TRACE"
fi
