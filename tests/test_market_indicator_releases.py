"""Actual, prior and pre-release-consensus provenance for macro indicators."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from tris_market_event_intelligence.market_evidence_db_bridge import make_packet_from_db
from tris_market_event_intelligence.market_indicator_releases import (
    IndicatorRelease,
    IndicatorReleaseBatch,
    IndicatorReleaseStore,
)


def sample() -> dict:
    return {
        "releaseId": "example-inflation-2024-02",
        "indicatorName": "Consumer inflation",
        "period": "2024-01",
        "publishedAt": "2024-02-13T08:30:00-05:00",
        "actual": "3.1%",
        "previous": "3.4%",
        "actualSourceName": "Official statistics agency",
        "actualSourceUrl": "https://www.bls.gov/news.release/cpi.htm",
        "actualExcerpt": (
            "Consumer inflation measured 3.1% year over year, compared "
            "with the previous month's 3.4% reading."
        ),
        "consensus": "2.9%",
        "consensusPublishedAt": "2024-02-12T15:00:00-05:00",
        "consensusSourceName": "Independent pre-release survey",
        "consensusSourceUrl": "https://www.example.org/survey",
        "consensusExcerpt": (
            "Economists surveyed ahead of release anticipated 2.9% year over year."
        ),
    }


def test_indicator_provenance_and_private_sqlite_bridge(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE articles (id INTEGER)")
    store = IndicatorReleaseStore(path)
    store.initialize()
    batch = IndicatorReleaseBatch(releases=[IndicatorRelease.model_validate(sample())])
    assert store.import_batch(batch) == {"inserted": 1, "unchanged": 0}
    assert store.import_batch(batch) == {"inserted": 0, "unchanged": 1}
    original = store.read(1)
    assert original.publishedAt.astimezone(UTC) == datetime.fromisoformat(
        "2024-02-13T13:30:00+00:00"
    )
    packet = make_packet_from_db(
        path, event_id="inflation", subject="Consumer inflation",
        as_of="2024-02-14", indicator_release_ids=(1,),
    )
    assert [x.sourceId for x in packet.evidences] == [
        "indicator-release-1-actual", "indicator-release-1-consensus",
    ]
    assert packet.evidences[0].indicatorValue == "3.1%"
    assert packet.evidences[0].indicatorPrevious == "3.4%"
    assert packet.evidences[1].expectation == "2.9%"
    assert all(x.verification == "REPORTED_UNVERIFIED" for x in packet.evidences)
    assert packet.evidences[0].publishedAt.astimezone(UTC).hour == 13


def test_consensus_cannot_be_added_after_release() -> None:
    row = sample()
    row["consensusPublishedAt"] = "2024-02-14T12:00:00+00:00"
    with pytest.raises(ValidationError, match="INDICATOR_CONSENSUS"):
        IndicatorRelease.model_validate(row)


def test_consensus_cannot_exist_without_independent_source_excerpt() -> None:
    row = sample()
    row["consensusExcerpt"] = "Market participants provided no number."
    with pytest.raises(ValidationError, match="INDICATOR_CONSENSUS"):
        IndicatorRelease.model_validate(row)


def test_indicator_release_is_immutable_and_checksum_checked(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    path.touch()
    store = IndicatorReleaseStore(path)
    store.initialize()
    batch = IndicatorReleaseBatch(releases=[IndicatorRelease.model_validate(sample())])
    store.import_batch(batch)
    edited = sample()
    edited["actual"] = "2.9%"
    edited["actualExcerpt"] = edited["actualExcerpt"].replace("3.1%", "2.9%")
    with pytest.raises(ValueError, match="IMMUTABLE"):
        store.import_batch(
            IndicatorReleaseBatch(releases=[IndicatorRelease.model_validate(edited)])
        )
    with sqlite3.connect(path) as db:
        db.execute("UPDATE market_indicator_releases SET payload_json='{}'")
    with pytest.raises(ValueError, match="CHECKSUM"):
        store.read(1)


def test_macro_release_batch_validation_preserves_true_source_claims() -> None:
    item = sample()
    del item["consensusSourceUrl"]
    with pytest.raises(ValidationError, match="INDICATOR_CONSENSUS"):
        IndicatorRelease.model_validate(item)
    duplicate = [sample(), sample()]
    with pytest.raises(ValidationError, match="INDICATOR_RELEASE_IDS_DUPLICATED"):
        IndicatorReleaseBatch.model_validate({
            "schemaVersion": "market_indicator_release_batch_v1",
            "releases": duplicate,
        })
