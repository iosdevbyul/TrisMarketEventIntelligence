"""No retrospective URL hint can become causal Gold or a sealed forecast."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tris_market_event_intelligence.historical_story_review import (
    prepare_historical_story_review,
    review_story_with_model,
)


def _fixture(tmp_path: Path) -> Path:
    def reference(url: str, slot: str) -> dict:
        return {"sourceUrl": url, "sourceName": "test.example",
                "gkgBatchSlotUtc": slot, "issuerVerified": False,
                "articleTextRetrieved": False, "independentCausalAttribution": False}
    archives = []
    for i, slot in enumerate((
        "2023-05-24T13:30:00+00:00",
        "2023-05-25T01:30:00+00:00",
        "2023-05-25T09:30:00+00:00",
        "2023-05-25T12:30:00+00:00",
    )):
        archives.append({
            "gkgSlot": slot, "archiveSlotNotPublicationTimestamp": True,
            "matchingDocumentReferences": [
                reference(
                    "https://publisher.example/nvidia-blowout-forecast-ai-rally"
                    if i == 0 else
                    "https://mirror.example/nvidia-blowout-forecast-ai-rally"
                    if i == 1 else
                    f"https://publisher.example/amd-radeon-chip-launch-details-{i}",
                    slot,
                ),
            ],
        })
    obj = {
        "schemaVersion": "gdelt_gkg_historical_scout_v1",
        "status": "UNVERIFIED_GDELT_DOCUMENT_REFERENCES",
        "ticker": "NVDA", "benchmark": "SPY",
        "shockDate": "2023-05-25", "shockRank": 1,
        "archivesFetched": 4, "archives": archives,
        "retrospectiveOutcomeKnown": True,
        "predictionOrTrainingPerformed": False,
        "sourceArticleRightsVerified": False,
    }
    source = tmp_path / "scout.json"
    source.write_text(json.dumps(obj))
    return source


class FakeProvider:
    model_id = "gemma4:e2b"

    def __init__(self, *, ids: list[str]) -> None:
        self.ids = ids
        self.calls = 0

    def generate_json(
        self, *, system: str, user: str, schema: dict[str, object] | None,
    ) -> str:
        self.calls += 1
        assert "NOT a price oracle" in system
        payload = json.loads(user)
        assert "stockReturn" not in user
        assert all(set(row) == {
            "referenceId", "urlDerivedHint", "candidateCategory", "gkgBatchSlotUtc"
        } for row in payload["urlOnlyEvidence"])
        assert schema is not None
        return json.dumps({
            "category": "COMPANY_EARNINGS_OR_GUIDANCE",
            "reference_ids": self.ids,
            "possible_mechanism": "Positive forward guidance could raise expected sales.",
            "limitations": "Only URL paths and dated company disclosure were supplied.",
            "source_uncertainty": "Actual original article release times are not known.",
            "causal_effect_proven": False,
        })


def test_historical_story_packet_has_primary_document_separate_from_gkg(
    tmp_path: Path,
) -> None:
    source = _fixture(tmp_path)
    report = prepare_historical_story_review(source)
    assert report["referenceCount"] == 4
    assert report["urlUniqueCount"] == 4
    assert report["candidateRepostGroupCount"] == 1
    assert report["officialPrimarySource"]["reportedRevenueUsdBillions"] == 7.19
    assert report["officialPrimarySource"]["outlookNextQuarterUsdBillions"] == 11.0
    assert report["officialPrimarySource"]["releaseClockIndependentlyVerified"] is False
    assert report["sourcePermissionsVerified"] is False
    assert report["actualMacroIndicatorsProvided"] is False
    assert report["priceCausalityVerified"] is False
    assert report["forwardPredictionSealed"] is False
    assert report["modelCalls"] == 0
    assert all(row["publisherFirstPublishedAt"] is None for row in report["references"])
    assert all(row["articleTextAvailable"] is False for row in report["references"])


def test_mock_gemma_reviews_urls_only_without_causal_promotion(tmp_path: Path) -> None:
    report = prepare_historical_story_review(_fixture(tmp_path))
    fake = FakeProvider(ids=["ref-001", "ref-002"])
    result = review_story_with_model(report, provider=fake)
    assert fake.calls == 1
    assert result["status"] == "MODEL_HYPOTHESIS_UNVERIFIED"
    assert result["hypothesis"]["reference_ids"] == ["ref-001", "ref-002"]
    assert result["causalAttributionVerified"] is False
    assert result["independentGoldCreated"] is False
    assert result["trainingPerformed"] is False
    assert result["forecastImprovementMeasured"] is False
    assert result["modelCalls"] == 1


def test_model_cannot_invent_reference_or_certify_causality(tmp_path: Path) -> None:
    report = prepare_historical_story_review(_fixture(tmp_path))
    with pytest.raises(ValueError, match="provenance"):
        review_story_with_model(report, provider=FakeProvider(ids=["ref-999"]))
    bad = FakeProvider(ids=["ref-001"])
    original = bad.generate_json

    def false_certainty(*, system: str, user: str, schema: dict | None) -> str:
        obj = json.loads(original(system=system, user=user, schema=schema))
        obj["causal_effect_proven"] = True
        return json.dumps(obj)

    bad.generate_json = false_certainty  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="contract"):
        review_story_with_model(report, provider=bad)


def test_missing_or_corrupted_source_is_rejected(tmp_path: Path) -> None:
    source = _fixture(tmp_path)
    doc = json.loads(source.read_text())
    doc["sourceArticleRightsVerified"] = True
    source.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="Expected unverified"):
        prepare_historical_story_review(source)


def test_non_earnings_event_has_no_automatic_official_disclosure(
    tmp_path: Path,
) -> None:
    source = _fixture(tmp_path)
    obj = json.loads(source.read_text())
    obj["shockDate"] = "2025-01-27"
    source.write_text(json.dumps(obj))
    report = prepare_historical_story_review(source)
    assert report["officialPrimarySource"] is None
    assert report["priceCausalityVerified"] is False
