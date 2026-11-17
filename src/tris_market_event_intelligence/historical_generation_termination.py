"""Diagnose generated historical prose completion without guessing decoder internals.

This is an observation-only classifier: a normal provider stop does not prove
the model completed its reasoning. Never rewrite or repair generated text.
"""
from __future__ import annotations

from typing import Any


def diagnose_termination(
    metadata: dict[str, Any] | None,
    *,
    narrative: str,
    schema_character_limit: int | None,
    json_valid: bool = True,
) -> dict[str, Any]:
    """Keep provider stop metadata distinct from observed narrative integrity."""
    data = metadata or {}
    finish = data.get("doneReason", data.get("finishReason"))
    configured_limit = data.get("configuredOutputTokenLimit")
    generated = data.get("generatedTokenCount")
    token_limit = (
        finish in {"length", "max_tokens"}
        or data.get("generatedOutputLimitReached") is True
    )
    boundary = (
        schema_character_limit is not None
        and len(narrative) == schema_character_limit
        and not narrative.rstrip().endswith((".", "!", "?"))
    )
    # A complete JSON object and a provider stop may coexist with a field
    # cut off by JSON Schema maxLength. Neither proves decoder internals.
    if token_limit:
        classification = "PROVIDER_OUTPUT_TOKEN_LIMIT"
    elif not json_valid:
        classification = "INVALID_JSON_OR_SCHEMA"
    elif boundary:
        classification = "SUSPECTED_SCHEMA_STRING_BOUNDARY"
    elif finish in {"stop", "eos", "end_turn"}:
        classification = "PROVIDER_NORMAL_STOP"
    elif finish is None:
        classification = "PROVIDER_TERMINATION_UNKNOWN"
    else:
        classification = "PROVIDER_OTHER_TERMINATION"
    return {
        "classification": classification,
        "providerTerminationReason": finish,
        "providerMetadataAvailable": bool(metadata),
        "generatedTokenCount": generated,
        "configuredOutputTokenLimit": configured_limit,
        "narrativeCharacterCount": len(narrative),
        "schemaCharacterLimit": schema_character_limit,
        "schemaBoundaryObserved": boundary,
        "jsonOrSchemaValid": json_valid,
        "decoderInternalCauseVerified": False,
        "semanticCompletionVerified": False,
        "modelTextRewritten": False,
    }
