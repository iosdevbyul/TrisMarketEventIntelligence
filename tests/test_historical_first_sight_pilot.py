"""First-sight integrated capture and model diagnostic tests (offline)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import httpx
import pytest

from tris_market_event_intelligence.historical_first_sight_pilot_cli import (
    capture,
    run_first_pass,
    validate_packet,
)


def packet() -> dict:
    return json.loads(Path(
        "research/independent-first-pass-pilot-v1.json"
    ).read_text(encoding="utf-8"))


class RecordingModel:
    model_id = "mock-unseen-evaluator"

    def __init__(self, bad_index: bool = False) -> None:
        self.prompts: list[dict] = []
        self.bad_index = bad_index

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        payload = json.loads(user)
        self.prompts.append(payload)
        assert payload["historicalPITVerified"] is False
        assert payload["goldAnswerAvailable"] is False
        assert "FTC" not in system or "FAA" not in system
        return json.dumps({
            "decisionStage": "PROPOSED",
            "implementationStatus": "NOT_OBSERVED",
            "evidenceStatus": "PARTIAL",
            "primaryExcerptIndex": 100 if self.bad_index else 0,
            "explanation": (
                "The cited official passage only documents the stated "
                "agency proposal at this cutoff."
            ),
            "limitation": (
                "The excerpt does not prove subsequent final approval "
                "or actual completed implementation."
            ),
        })


def _client(p: dict, *, missing: bool = False) -> httpx.Client:
    sources = {
        source["url"]: source
        for event in p["events"]
        for source in event["sources"]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        source = sources[str(request.url)]
        pieces = [x["quote"] for x in source["excerpts"]]
        if missing:
            pieces.pop()
        # An explicit post-cutoff injected banner must never flow into
        # curated model prompts, even though it appears in the live HTML.
        page = (
            "<html><body><header>Future judicial decisions in 2025</header>"
            + " ".join(pieces)
            + "</body></html>"
        )
        return httpx.Response(
            200, text=page, headers={"content-type": "text/html"},
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_capture_then_first_pass_does_not_leak_future_pages_or_gold() -> None:
    p = packet()
    with _client(p) as client:
        recorded = capture(p, client=client)
    assert len(recorded["sources"]) == 4
    assert recorded["historicalPITVerified"] is False
    assert all(s["historicalSnapshotVerified"] is False
               for s in recorded["sources"])
    provider = RecordingModel()
    result = run_first_pass(p, recorded, provider)
    assert result["modelCalls"] == 8
    assert len(provider.prompts) == 8
    assert result["trustEligible"] is False
    assert result["generalizationVerified"] is False
    early = provider.prompts[0]["visibleOfficialExcerpts"]
    assert all(s["sourceDate"] == "2023-01-05" for s in early)
    assert all(
        "Future judicial decisions" not in json.dumps(x)
        for x in provider.prompts
    )
    assert all(
        "reference" not in json.dumps(x).lower()
        for x in provider.prompts
    )
    assert result["results"][0]["primaryEvidence"]["claimId"] == "proposal"


def test_capture_rejects_missing_verbatim_excerpt() -> None:
    with _client(packet(), missing=True) as client:
        with pytest.raises(ValueError, match="FIRST_SIGHT_EXCERPT_NOT_FOUND"):
            capture(packet(), client=client)


def test_inference_rejects_tampered_capture_or_model_index() -> None:
    p = packet()
    with _client(p) as client:
        recorded = capture(p, client=client)
    tampered = copy.deepcopy(recorded)
    tampered["packetSha256"] = "fake"
    with pytest.raises(ValueError, match="FIRST_SIGHT_CAPTURE_MISMATCH"):
        run_first_pass(p, tampered, RecordingModel())
    with pytest.raises(ValueError, match="FIRST_SIGHT_PRIMARY_INDEX_INVALID"):
        run_first_pass(p, recorded, RecordingModel(bad_index=True))


def test_invalid_new_sources_cannot_be_admitted() -> None:
    p = packet()
    p["events"][0]["sources"][0]["url"] = (
        "https://attacker.example/pretending-to-be-ftc"
    )
    with pytest.raises(ValueError, match="FIRST_SIGHT_SOURCE_INVALID"):
        validate_packet(p)
    p = packet()
    p["historicalPITVerified"] = True
    with pytest.raises(ValueError, match="FIRST_SIGHT_PACKET_INVALID"):
        validate_packet(p)
