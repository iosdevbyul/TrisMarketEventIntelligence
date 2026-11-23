"""Regression tests for the Gemma-to-real-price outcome evaluation bridge."""
import csv
import hashlib
import json
from pathlib import Path

import pytest

from tris_market_event_intelligence.market_outcome_validation_cli import evaluate


def _csv(path: Path, fields: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def _fixture(tmp_path: Path):
    folder = tmp_path / "gemma"
    folder.mkdir()
    body = {
        "schemaVersion": "gemma_original_article_review_v2",
        "model": "gemma4:e2b",
        "processedAt": "2026-01-05T20:00:00+00:00",
        "article": {
            "articleId": 1, "title": "Example", "publishedAt": "2026-01-05T15:00:00+00:00",
        },
        "analysis": {
            "impacts": [{
                "target": "Example Corp", "direction": "UP",
                "reason": "Demand expected to increase", "evidence_quote": "Original text",
            }, {
                "target": "Unknown", "direction": "DOWN",
                "reason": "Possible risk", "evidence_quote": "Original text",
            }],
        },
        "predictionSealedBeforeOutcome": False,
    }
    (folder / "1.json").write_text(json.dumps(body), encoding="utf-8")
    mappings = tmp_path / "mappings.csv"
    _csv(mappings, ["target", "market", "symbol", "benchmark", "verified_at"], [
        ["Example Corp", "XNAS", "EXMP", "SPY", "2026-01-01T00:00:00+00:00"],
    ])
    prices = tmp_path / "prices.csv"
    _csv(prices, ["symbol", "date", "adjusted_close", "source"], [
        ["EXMP", "2026-01-06", "100", "test sourced fixture"],
        ["EXMP", "2026-01-07", "103", "test sourced fixture"],
        ["EXMP", "2026-01-08", "104", "test sourced fixture"],
        ["EXMP", "2026-01-09", "105", "test sourced fixture"],
        ["EXMP", "2026-01-12", "106", "test sourced fixture"],
        ["EXMP", "2026-01-13", "107", "test sourced fixture"],
        ["SPY", "2026-01-06", "200", "test sourced fixture"],
        ["SPY", "2026-01-07", "201", "test sourced fixture"],
        ["SPY", "2026-01-08", "202", "test sourced fixture"],
        ["SPY", "2026-01-09", "203", "test sourced fixture"],
        ["SPY", "2026-01-12", "204", "test sourced fixture"],
        ["SPY", "2026-01-13", "205", "test sourced fixture"],
    ])
    return folder, mappings, prices


def test_predictions_join_to_prices_and_preserve_unmapped(tmp_path):
    folder, mappings, prices = _fixture(tmp_path)
    result = evaluate(folder, mappings, prices, price_source_description="fixture")
    assert result["articlesInspected"] == 1
    assert result["impactsInspected"] == 2
    assert result["metrics"]["1d"]["correct"] == 1
    assert result["metrics"]["5d"]["correct"] == 1
    assert result["metrics"]["1d"]["accuracy"] == 1.0
    item, unmapped = result["items"]
    assert item["entryDay"] == "2026-01-06"
    assert item["evaluationClass"] == "RETROSPECTIVE_UNSEALED"
    assert not item["predictionSealedBeforeOutcome"]
    assert item["horizons"][0]["stockReturn"] == pytest.approx(0.03)
    assert item["horizons"][0]["benchmarkReturn"] == pytest.approx(0.005)
    assert item["horizons"][0]["excessReturn"] == pytest.approx(0.025)
    assert item["predictionSourceSha256"] == hashlib.sha256(
        (folder / "1.json").read_bytes()
    ).hexdigest()
    assert unmapped["status"] == "UNMAPPED"


def test_later_verified_mapping_is_explicitly_retrospective(tmp_path):
    folder, mappings, prices = _fixture(tmp_path)
    _csv(mappings, ["target", "market", "symbol", "benchmark", "verified_at"], [
        ["Example Corp", "XNAS", "EXMP", "SPY", "2026-02-01T00:00:00+00:00"],
    ])
    result = evaluate(folder, mappings, prices, price_source_description="fixture")
    assert result["items"][0]["status"] == "COMPLETED"
    assert result["items"][0]["mappingPointInTime"] is False
    assert result["items"][0]["evaluationClass"] == "RETROSPECTIVE_POST_EVENT_MAPPING"
    assert result["metrics"]["1d"]["accuracy"] == 1.0


def test_prices_must_be_provenanced_and_positive(tmp_path):
    folder, mappings, prices = _fixture(tmp_path)
    _csv(prices, ["symbol", "date", "adjusted_close", "source"], [
        ["EXMP", "2026-01-06", "0", ""],
    ])
    with pytest.raises(ValueError):
        evaluate(folder, mappings, prices, price_source_description="fixture")


def test_duplicate_article_ids_rejected(tmp_path):
    folder, mappings, prices = _fixture(tmp_path)
    (folder / "duplicate.json").write_bytes((folder / "1.json").read_bytes())
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate(folder, mappings, prices, price_source_description="fixture")


def _operational_db_for_fixture(tmp_path: Path, folder: Path) -> Path:
    import sqlite3

    from test_operational_news_pit_readiness_cli import create_db

    db_path = tmp_path / "news.sqlite3"
    create_db(db_path)
    item_path = folder / "1.json"
    gemma = json.loads(item_path.read_text(encoding="utf-8"))
    with sqlite3.connect(db_path) as db:
        db.execute(
            "UPDATE articles SET published_at=?, detected_at=? WHERE id=1",
            ("2026-01-05T15:00:00+00:00", "2026-01-05T15:30:00+00:00"),
        )
        db.execute("UPDATE captures SET fetched_at=? WHERE id=1",
                   ("2026-01-05T15:30:00+00:00",))
        db.execute("UPDATE article_full_text SET observed_at=? WHERE article_id=1",
                   ("2026-01-05T15:45:00+00:00",))
        digest = db.execute(
            "SELECT body_sha256 FROM article_full_text WHERE article_id=1"
        ).fetchone()[0]
    gemma["article"].update({
        "url": "https://example.org/article",
        "detectedAt": "2026-01-05T15:30:00+00:00",
        "bodySha256": digest,
    })
    item_path.write_text(json.dumps(gemma), encoding="utf-8")
    return db_path


def test_real_db_identity_gates_daily_market_outcome(tmp_path: Path) -> None:
    folder, mappings, prices = _fixture(tmp_path)
    news_db = _operational_db_for_fixture(tmp_path, folder)
    before = news_db.read_bytes()
    report = evaluate(
        folder, mappings, prices, news_db=news_db,
        price_source_description="synthetic comparison",
    )
    assert news_db.read_bytes() == before
    assert report["operationalNewsDatabaseAudited"] is True
    assert report["items"][0]["sourceAuditStatus"] == "STORED_ORIGINAL_SHA_MATCHED"
    assert report["items"][0]["horizons"][0]["dataStatus"] == "COMPLETED"
    assert report["prospectivePredictionAccuracy"] is None
    assert report["trainingEligible"] is False
    assert report["constantDirectionBaselines"]["1d"]["denominator"] == 1
    assert report["constantDirectionBaselines"]["1d"]["alwaysUp"] == 1


def test_mismatched_gemma_digest_withholds_daily_score(tmp_path: Path) -> None:
    folder, mappings, prices = _fixture(tmp_path)
    news_db = _operational_db_for_fixture(tmp_path, folder)
    article_path = folder / "1.json"
    item = json.loads(article_path.read_text())
    item["article"]["bodySha256"] = "0" * 64
    article_path.write_text(json.dumps(item), encoding="utf-8")
    result = evaluate(
        folder, mappings, prices, news_db=news_db,
        price_source_description="synthetic comparison",
    )
    target = result["items"][0]
    assert target["status"] == "NEWS_SOURCE_NOT_ADMITTED"
    assert target["sourceAuditStatus"] == "NEWS_SOURCE_IDENTITY_MISMATCH"
    assert target["horizons"] == []
    assert result["metrics"]["1d"]["denominator"] == 0
    assert result["metrics"]["1d"]["accuracy"] is None
    assert result["constantDirectionBaselines"]["1d"]["denominator"] == 0


def test_unsealed_reporting_is_not_prospective_accuracy(tmp_path: Path) -> None:
    folder, mappings, prices = _fixture(tmp_path)
    result = evaluate(
        folder, mappings, prices,
        price_source_description="synthetic comparison",
    )
    assert result["marketReturnObservedFromSourcedDailyClose"] is True
    assert result["prospectivePredictionAccuracy"] is None
    assert result["predictionSealAuthenticated"] is False
    assert result["priceProviderIndependentlyAuthenticated"] is False
    assert result["items"][0]["evaluationClass"] == "RETROSPECTIVE_UNSEALED"


def test_recovery_manifest_must_match_actual_historical_price_bytes(tmp_path: Path) -> None:
    folder, mappings, prices = _fixture(tmp_path)
    manifest = tmp_path / "prices.csv.manifest.json"
    manifest.write_text(json.dumps({
        "schemaVersion": "market_price_recovery_v1",
        "outputSha256": hashlib.sha256(prices.read_bytes()).hexdigest(),
        "mappingsSha256": hashlib.sha256(mappings.read_bytes()).hexdigest(),
        "completeDateCoverageGuaranteed": False,
    }), encoding="utf-8")
    report = evaluate(
        folder, mappings, prices, price_source_description="synthetic comparison",
        price_manifest=manifest,
    )
    assert report["priceRecoveryManifestVerified"] is True
    assert report["metrics"]["1d"]["distinctArticlesScored"] == 1
    assert report["metrics"]["1d"]["distinctPublicationDatesScored"] == 1
    assert report["metrics"]["1d"]["distinctStoryCandidateGroupsScored"] == 0
    manifest.write_text(json.dumps({
        "outputSha256": "0" * 64,
        "mappingsSha256": hashlib.sha256(mappings.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA mismatch"):
        evaluate(
            folder, mappings, prices,
            price_source_description="synthetic comparison",
            price_manifest=manifest,
        )
