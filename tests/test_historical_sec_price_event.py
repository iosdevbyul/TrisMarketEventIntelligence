"""SEC acceptance time becomes point-in-time source evidence, not causal Gold."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tris_market_event_intelligence.historical_sec_price_event_cli import (
    build_sec_earnings_price_event,
)

SEC = Path("research/nvda-2018-11-15-sec-8k-acceptance.json")
PRIMARY = Path("research/nvda-2018-11-15-issuer-primary-evidence.json")


def _inputs(tmp_path: Path) -> tuple[dict, dict, Path, Path]:
    issuer = json.loads(PRIMARY.read_text())
    sec = json.loads(SEC.read_text())
    corroboration = {
        "schemaVersion": "historical_source_corrob_review_v1",
        "status": "OFFICIAL_DATE_AND_FACTS_CORROBORATED_CLOCK_UNVERIFIED",
        "eventId": issuer["eventId"],
        "ticker": "NVDA",
        "shockDate": "2018-11-16",
        "priceImpactCausalityVerified": False,
        "replayTrainingEligible": False,
        "gkgCandidateCount": 21,
        "issuerSource": {
            "publisherFirstPublicationClockVerified": False,
            "facts": issuer["facts"],
        },
    }
    stock = tmp_path / "nvda.csv"
    spy = tmp_path / "spy.csv"
    stock.write_text(
        "date,adjusted_close,volume\n"
        "2018-11-13,12,100\n2018-11-14,11,100\n"
        "2018-11-15,10,100\n2018-11-16,8,100\n"
        "2018-11-19,9,100\n"
    )
    spy.write_text(
        "date,adjusted_close,volume\n"
        "2018-11-13,100,100\n2018-11-14,101,100\n"
        "2018-11-15,100,100\n2018-11-16,99,100\n"
        "2018-11-19,100,100\n"
    )
    return corroboration, sec, stock, spy


def _run(tmp_path: Path) -> dict:
    packet, sec, stock, spy = _inputs(tmp_path)
    return build_sec_earnings_price_event(
        corroboration=packet, corroboration_sha="a" * 64,
        sec=sec, sec_sha="b" * 64, stock_csv=stock,
        benchmark_csv=spy,
    )


def test_genuine_sec_acceptance_is_before_next_session_cutoff(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result["status"] == "SEC_FILING_PREOPEN_PRICE_OUTCOME_OBSERVED_RETROSPECTIVE"
    assert result["decisionAt"] == "2018-11-16T14:30:00+00:00"
    assert result["features"]["secAcceptedAt"] == "2018-11-15T21:21:48+00:00"
    assert result["features"]["issuerDocumentAvailableAsSecExhibit"] is True
    assert len(result["features"]["issuerFacts"]) == 3
    assert result["features"]["gkgCandidateUrlsExcluded"] == 21
    assert result["features"]["publisherFirstReleaseTimeVerified"] is False
    assert result["outcome"]["stockReturn"] == pytest.approx(-0.2)
    assert result["outcome"]["benchmarkReturn"] == pytest.approx(-0.01)
    assert result["outcome"]["excessReturn"] == pytest.approx(-0.19)
    assert result["trainingReady"] is False
    assert result["modelPredictionMadeBeforeOutcome"] is False
    assert result["causalImpactVerified"] is False


def test_later_sec_timestamp_fails_closed(tmp_path: Path) -> None:
    packet, sec, stock, spy = _inputs(tmp_path)
    sec["form8kAcceptedEastern"] = "2018-11-16T09:31:00-05:00"
    with pytest.raises(ValueError, match="SEC acceptance clock"):
        build_sec_earnings_price_event(
            corroboration=packet, corroboration_sha="a" * 64, sec=sec,
            sec_sha="b" * 64, stock_csv=stock, benchmark_csv=spy,
        )


def test_cannot_pretend_sec_clock_is_publishers_first_timestamp(tmp_path: Path) -> None:
    packet, sec, stock, spy = _inputs(tmp_path)
    sec["secAcceptanceIsPublisherInitialRelease"] = True
    with pytest.raises(ValueError, match="SEC first-publication"):
        build_sec_earnings_price_event(
            corroboration=packet, corroboration_sha="a" * 64, sec=sec,
            sec_sha="b" * 64, stock_csv=stock, benchmark_csv=spy,
        )


def test_price_outcome_requires_both_matching_sessions(tmp_path: Path) -> None:
    packet, sec, stock, spy = _inputs(tmp_path)
    spy.write_text(spy.read_text().replace("2018-11-15,100,100\n", ""))
    with pytest.raises(ValueError, match="calendars"):
        build_sec_earnings_price_event(
            corroboration=packet, corroboration_sha="a" * 64, sec=sec,
            sec_sha="b" * 64, stock_csv=stock, benchmark_csv=spy,
        )


def test_event_mismatch_rejects_accidental_other_issuer(tmp_path: Path) -> None:
    packet, sec, stock, spy = _inputs(tmp_path)
    packet["eventId"] = "UNRELATED-EVENT"
    with pytest.raises(ValueError, match="SEC first-publication"):
        build_sec_earnings_price_event(
            corroboration=packet, corroboration_sha="a" * 64, sec=sec,
            sec_sha="b" * 64, stock_csv=stock, benchmark_csv=spy,
        )
