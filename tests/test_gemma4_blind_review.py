"""Blind Gemma pilot regression: per-event masking, source evidence, human-only gold."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tris_market_event_intelligence import gemma4_blind_review
from tris_market_event_intelligence.gemma4_blind_review import (
    _hash,
    audit_blind_judgments,
    build_blind_packet,
    prepare_blind_judgments,
)
from tris_market_event_intelligence.gemma4_blind_review_cli import (
    _create_private,
    _read_private,
)
from tris_market_event_intelligence.impact import (
    EventInput,
    Impact,
    ImpactAnalysisError,
    ImpactResult,
)


def article() -> EventInput:
    return EventInput(
        17, "ECONOMIC_INDICATOR", "UNVERIFIED", None, None,
        "Consumer Price Index increased during the latest reporting period.",
        "local-news", "https://example.org/17",
        "2026-11-15T00:00:00Z", "REPORTED",
    )


def result() -> ImpactResult:
    return ImpactResult(impacts=[Impact(
        market="US", sector="FINANCIALS", direction="UNCERTAIN",
        horizon="SHORT_TERM",
        rationale=(
            "If the reported inflation increase affects future borrowing costs, "
            "bank valuations may change; expectations and verification are unknown."
        ),
        evidence_excerpt="Consumer Price Index increased",
        uncertainty="HIGH", drivers=[], entities=[],
    )])


class Stub:
    def __init__(self, model: str, *, fails: bool = False) -> None:
        self.model_id = model
        self.fails = fails
        self.inputs: list[EventInput] = []

    def analyze(self, entry: EventInput) -> ImpactResult:
        self.inputs.append(entry)
        if self.fails:
            raise ImpactAnalysisError("EVIDENCE_MISMATCH", "Unsafe source quoted")
        return result()


def generate(monkeypatch, tmp_path: Path, reverse: bool = False):
    monkeypatch.setattr(
        gemma4_blind_review, "select_news_events",
        lambda path, **kwargs: [article()],
    )
    baseline = Stub("gemma2:latest")
    challenger = Stub("gemma4:e4b", fails=True)
    packet, mapping = build_blind_packet(
        db_path=tmp_path / "unused.sqlite3", analyzers=(baseline, challenger),
        choose_first=lambda: reverse,
    )
    return packet, mapping, baseline, challenger


def complete(packet):
    judgments = prepare_blind_judgments(packet)
    row = judgments["items"][0]["humanReview"]
    row["newsAttribution"] = "PARTIAL"
    for slot in ("A", "B"):
        for dimension in (
            "evidenceAccuracy", "contextAndMechanism", "uncertaintyHonesty",
        ):
            row[slot][dimension] = "SUPPORTED" if slot == "A" else "UNASSESSABLE"
    row["preferredResponse"] = "A"
    row["reviewReason"] = (
        "Original source only supports a reported claim; the other answer "
        "failed the evidence contract and cannot be compared substantively."
    )
    judgments["humanReviewCompleted"] = True
    return judgments


def test_blind_packet_no_model_names_and_sealed_assignment(monkeypatch, tmp_path: Path):
    packet, mapping, baseline, challenger = generate(monkeypatch, tmp_path)
    assert baseline.inputs == challenger.inputs == [article()]
    assert "gemma2" not in json.dumps(packet).lower()
    assert "gemma4" not in json.dumps(packet).lower()
    assert packet["items"][0]["responses"]["A"]["status"] == "VALIDATED_OUTPUT"
    assert packet["items"][0]["responses"]["B"]["status"] == "REJECTED_CONTRACT"
    assert mapping["assignments"][0]["A"] == "gemma2:latest"
    assert mapping["assignments"][0]["B"] == "gemma4:e4b"
    assert mapping["packetSha256"] == _hash(packet)
    assert packet["humanNewsQualityVerified"] is False
    assert packet["promotionAllowed"] is False
    assert packet["stockPricesUsed"] is False


def test_assignment_can_be_reversed_without_rewriting_model_response(
    monkeypatch, tmp_path: Path,
):
    packet, mapping, _, _ = generate(monkeypatch, tmp_path, reverse=True)
    assert mapping["assignments"][0]["A"] == "gemma4:e4b"
    assert mapping["assignments"][0]["B"] == "gemma2:latest"
    assert packet["items"][0]["responses"]["A"]["status"] == "REJECTED_CONTRACT"
    assert packet["items"][0]["responses"]["B"]["status"] == "VALIDATED_OUTPUT"


def test_review_requires_real_human_input_not_inferred_gold(monkeypatch, tmp_path: Path):
    packet, _, _, _ = generate(monkeypatch, tmp_path)
    empty = prepare_blind_judgments(packet)
    with pytest.raises(ValueError, match="grade"):
        audit_blind_judgments(packet, empty, reviewer="reviewer-1")
    reviewed = complete(packet)
    report = audit_blind_judgments(packet, reviewed, reviewer="reviewer-1")
    assert report["status"] == "HUMAN_JUDGMENTS_FORMAT_VERIFIED_NOT_GOLD_CERTIFIED"
    assert report["blindPreferences"] == {"A": 1}
    assert report["modelIdentitiesDisclosed"] is False
    assert report["modelQualityImprovementValidated"] is False
    assert report["promotionAllowed"] is False
    assert "gemma4:e4b" not in json.dumps(report)


def test_tampered_news_packet_hash_fails_closed(monkeypatch, tmp_path: Path):
    packet, _, _, _ = generate(monkeypatch, tmp_path)
    reviewed = complete(packet)
    packet["items"][0]["sourceContext"]["evidence"] = "Altered input"
    with pytest.raises(ValueError, match="different blind packet"):
        audit_blind_judgments(packet, reviewed, reviewer="reviewer-1")


def test_missing_or_reordered_review_refused(monkeypatch, tmp_path: Path):
    packet, _, _, _ = generate(monkeypatch, tmp_path)
    reviewed = complete(packet)
    reviewed["items"][0]["eventId"] = 999
    with pytest.raises(ValueError, match="misaligned"):
        audit_blind_judgments(packet, reviewed, reviewer="reviewer-1")


def test_invalid_rubric_is_rejected(monkeypatch, tmp_path: Path):
    packet, _, _, _ = generate(monkeypatch, tmp_path)
    reviewed = complete(packet)
    reviewed["items"][0]["humanReview"]["A"]["evidenceAccuracy"] = "PERFECT"
    with pytest.raises(ValueError, match="ratings"):
        audit_blind_judgments(packet, reviewed, reviewer="reviewer-1")


def test_private_output_owner_only_and_create_only(monkeypatch, tmp_path: Path):
    packet, mapping, _, _ = generate(monkeypatch, tmp_path)
    p = tmp_path / "packet.json"
    m = tmp_path / "sealed-mapping.json"
    _create_private(p, packet)
    _create_private(m, mapping)
    assert p.stat().st_mode & 0o077 == 0
    assert m.stat().st_mode & 0o077 == 0
    assert _read_private(p) == packet
    assert _read_private(m) == mapping
    with pytest.raises(ValueError, match="Create-only"):
        _create_private(p, packet)


def test_empty_archive_no_fake_review(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        gemma4_blind_review, "select_news_events",
        lambda path, **kwargs: [],
    )
    packet, key = build_blind_packet(
        db_path=tmp_path / "none.sqlite3",
        analyzers=(Stub("gemma2:latest"), Stub("gemma4:e4b")),
    )
    assert packet["items"] == []
    assert key["assignments"] == []
    with pytest.raises(ValueError, match="No model responses"):
        prepare_blind_judgments(packet)


def test_incorrect_model_identity_or_invalid_limit_fails_before_selection(
    tmp_path: Path,
):
    with pytest.raises(ValueError, match="identities"):
        build_blind_packet(
            db_path=tmp_path / "none", analyzers=(
                Stub("gemma2:latest"), Stub("gemma4:12b"),
            ),
        )
    with pytest.raises(ValueError, match="1..3"):
        build_blind_packet(
            db_path=tmp_path / "none", analyzers=(
                Stub("gemma2:latest"), Stub("gemma4:e4b"),
            ), limit=0,
        )
