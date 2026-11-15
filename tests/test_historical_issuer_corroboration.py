"""Original issuer earnings can contextualize GKG hints without falsifying clocks."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tris_market_event_intelligence.historical_issuer_corroboration_cli import (
    _write,
    corroborate_sources,
)

ISSUER = Path("research/nvda-2018-11-15-issuer-primary-evidence.json")


def _source() -> dict:
    return json.loads(ISSUER.read_text(encoding="utf-8"))


def _review() -> dict:
    references = [
        {
            "referenceId": "ref-001",
            "sourceUrl": "https://publisher.example/crypto-gpu-guidance",
            "candidateCategory": "COMPANY_EARNINGS_OR_GUIDANCE",
            "articleTextAvailable": False,
            "publisherFirstPublishedAt": None,
            "earliestTradableAvailabilityVerified": False,
            "issuerMatchIndependentlyVerified": False,
        },
        {
            "referenceId": "ref-002",
            "sourceUrl": "https://publisher.example/general-sector-review",
            "candidateCategory": "COMPETITOR_OR_SECTOR",
            "articleTextAvailable": False,
            "publisherFirstPublishedAt": None,
            "earliestTradableAvailabilityVerified": False,
            "issuerMatchIndependentlyVerified": False,
        },
    ]
    return {
        "schemaVersion": "historical_story_review_v1",
        "status": "URL_METADATA_ONLY_REVIEW_REQUIRED",
        "ticker": "NVDA",
        "shockDate": "2018-11-16",
        "retrospectiveOutcomeKnown": True,
        "notIndependentGold": True,
        "articleTextsFetched": 0,
        "priceCausalityVerified": False,
        "promotionAllowed": False,
        "referenceCount": 2,
        "references": references,
        "candidateStoryGroups": [{
            "referenceIds": ["ref-001", "ref-002"],
            "sameUnderlyingStoryVerified": False,
        }],
    }


def test_real_issuer_data_source_has_no_invented_first_publish_clock() -> None:
    source = _source()
    assert source["primarySource"]["announcementDate"] == "2018-11-15"
    assert source["primarySource"]["publisherFirstPublicationTimeUtc"] is None
    assert source["primarySource"]["sourceType"] == "ISSUER_ORIGINAL_PRESS_RELEASE"
    assert source["truthControls"]["sufficientForProspectiveTraining"] is False
    assert len(source["facts"]) == 3


def test_corroboration_preserves_uncertainty_and_separates_sources() -> None:
    result = corroborate_sources(
        review=_review(), review_sha="a" * 64,
        issuer=_source(), issuer_sha="b" * 64,
    )
    assert result["gkgCandidateCount"] == 2
    assert result["gkgOfficialExactUrlMatchRefs"] == []
    assert result["issuerSource"]["verifiedDirectIssuerAssociation"] is True
    assert result["issuerSource"]["publisherFirstPublicationClockVerified"] is False
    assert result["gkgCandidates"][0]["usableAtHistoricalPredictionCutoff"] is False
    assert result["replayTrainingEligible"] is False
    assert result["priceImpactCausalityVerified"] is False
    assert result["modelCalls"] == 0


def test_forged_publication_clock_and_gkg_source_proof_fail_closed() -> None:
    doc = _source()
    doc["primarySource"]["publisherFirstPublicationTimeUtc"] = (
        "2018-11-15T21:00:00+00:00"
    )
    with pytest.raises(ValueError, match="publication-time"):
        corroborate_sources(
            review=_review(), review_sha="a" * 64,
            issuer=doc, issuer_sha="b" * 64,
        )
    bad = _review()
    bad["references"][0]["issuerMatchIndependentlyVerified"] = True
    with pytest.raises(ValueError, match="GKG metadata"):
        corroborate_sources(
            review=bad, review_sha="a" * 64,
            issuer=_source(), issuer_sha="b" * 64,
        )


def test_untrusted_group_rejected_and_output_is_create_only(tmp_path: Path) -> None:
    review = _review()
    review["candidateStoryGroups"][0]["sameUnderlyingStoryVerified"] = True
    with pytest.raises(ValueError, match="heuristic"):
        corroborate_sources(
            review=review, review_sha="a" * 64,
            issuer=_source(), issuer_sha="b" * 64,
        )
    safe = corroborate_sources(
        review=_review(), review_sha="a" * 64,
        issuer=_source(), issuer_sha="b" * 64,
    )
    output = tmp_path / "event-research.json"
    _write(output, safe)
    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="new"):
        _write(output, copy.deepcopy(safe))
