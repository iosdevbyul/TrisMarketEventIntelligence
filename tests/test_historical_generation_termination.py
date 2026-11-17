"""Regression coverage for model stop attribution (no model execution)."""
from __future__ import annotations

from tris_market_event_intelligence.historical_generation_termination import (
    diagnose_termination,
)


def test_provider_stop_is_not_semantic_completion() -> None:
    report = diagnose_termination(
        {"doneReason": "stop", "generatedTokenCount": 183,
         "configuredOutputTokenLimit": 4096},
        narrative="The FAA formally approved inspection procedures.",
        schema_character_limit=1200,
    )
    assert report["classification"] == "PROVIDER_NORMAL_STOP"
    assert report["semanticCompletionVerified"] is False


def test_exact_schema_boundary_outweighs_normal_stop() -> None:
    report = diagnose_termination(
        {"doneReason": "stop", "generatedTokenCount": 192,
         "configuredOutputTokenLimit": 4096},
        narrative=("The FAA described an official action. " * 20)[:600].rstrip().ljust(600, "a"),
        schema_character_limit=600,
    )
    assert report["classification"] == "SUSPECTED_SCHEMA_STRING_BOUNDARY"
    assert report["decoderInternalCauseVerified"] is False


def test_token_limit_and_invalid_json_are_distinct() -> None:
    limited = diagnose_termination(
        {"doneReason": "length", "generatedOutputLimitReached": True},
        narrative="The FAA was", schema_character_limit=1200,
        json_valid=False,
    )
    malformed = diagnose_termination(
        {"doneReason": "stop"}, narrative="{",
        schema_character_limit=1200, json_valid=False,
    )
    assert limited["classification"] == "PROVIDER_OUTPUT_TOKEN_LIMIT"
    assert malformed["classification"] == "INVALID_JSON_OR_SCHEMA"


def test_unknown_provider_and_full_sentence_at_boundary() -> None:
    unknown = diagnose_termination(
        None, narrative="The approval status is unknown.",
        schema_character_limit=1200,
    )
    complete = diagnose_termination(
        {"doneReason": "stop"}, narrative="A full sentence.",
        schema_character_limit=len("A full sentence."),
    )
    assert unknown["classification"] == "PROVIDER_TERMINATION_UNKNOWN"
    assert complete["schemaBoundaryObserved"] is False
