"""Independent reserved chronology and audit-only model validation."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from tris_market_event_intelligence import sequential_holdout_cli as holdout


def test_frozen_holdout_excludes_training_articles_and_pretraining_dates(
    monkeypatch, tmp_path: Path,
):
    db_path = tmp_path / "news.sqlite3"
    db_path.touch()
    journal = tmp_path / "study.sqlite3"
    settings = {
        "sourceDb": str(db_path.resolve()),
        "sourceContentCache": None,
        "cutoff": "2026-11-19T00:00:00+00:00",
        "horizonHours": 24,
        "interval": "1d",
        "model": "local-mock",
    }
    with sqlite3.connect(journal) as db:
        db.execute("CREATE TABLE runs(id INTEGER, settings_json TEXT)")
        db.execute("INSERT INTO runs VALUES (1,?)", (json.dumps(settings),))
        db.execute(
            "CREATE TABLE steps(article_id INTEGER, observed_at TEXT, "
            "phase TEXT, prediction_json TEXT, evaluation_json TEXT)"
        )
        db.execute(
            "INSERT INTO steps VALUES (?, ?, ?, ?, ?)",
            (1, "2024-01-01T00:00:00+00:00", "COMPLETE",
             json.dumps({"impacts": []}), json.dumps({"impacts": []})),
        )
    rows = [
        {
            "articleId": 1, "publishedAtReported": "2024-01-02T00:00:00+00:00",
            "title": "training duplicate", "summary": "body",
            "source": "industry_rss", "firstObservedAt": "2024-01-02T00:00:00+00:00",
        },
        {
            "articleId": 2, "publishedAtReported": "2023-12-31T00:00:00+00:00",
            "title": "earlier", "summary": "body",
            "source": "industry_rss", "firstObservedAt": "2023-12-31T00:00:00+00:00",
        },
        {
            "articleId": 3, "publishedAtReported": "2024-03-01T00:00:00+00:00",
            "title": "NVIDIA raises guidance", "summary":
            "NVIDIA raised its earnings outlook after stronger sales.",
            "source": "industry_rss", "firstObservedAt": "2024-03-01T00:00:00+00:00",
        },
    ]
    monkeypatch.setattr(holdout, "load_articles",
                        lambda *a, **k: (rows, 3))
    monkeypatch.setattr(
        holdout, "pre_event_market_context",
        lambda *a: {"status": "UNKNOWN"},
    )
    monkeypatch.setattr(
        holdout, "predict_with_repair",
        lambda *a: {"summary": "NVIDIA earnings guidance", "impacts": []},
    )
    monkeypatch.setattr(
        holdout, "validate_asset_evidence", lambda prediction, article: prediction,
    )
    path = tmp_path / "holdout.json"
    result = holdout.evaluate_holdout(
        journal=journal, source_db=db_path, output=path,
        model=object(), client=object(),
    )
    assert result["holdoutSelected"] == 1
    assert result["holdoutCompleted"] == 1
    assert result["directionalPriceScored"] == 0
    assert result["learningDisabled"] is True
    assert result["pointInTimeCertified"] is False
    saved = json.loads(path.read_text())
    assert [row["articleId"] for row in saved["articles"]] == [3]
    with pytest.raises(ValueError, match="overwrite"):
        holdout.evaluate_holdout(
            journal=journal, source_db=db_path, output=path,
            model=object(), client=object(),
        )


def test_price_direction_controls_use_exact_same_scored_assets():
    cases = [
        {"evaluation": {"impacts": [
            {"status": "WRONG", "direction": "NEUTRAL",
             "actualDirection": "POSITIVE", "relativeDirectionVerdict": "WRONG"},
            {"status": "CORRECT", "direction": "NEGATIVE",
             "actualDirection": "NEGATIVE",
             "relativeDirectionVerdict": "CORRECT"},
        ]}},
        {"evaluation": {"impacts": [
            {"status": "UNVERIFIED", "direction": "POSITIVE"},
            {"status": "WRONG", "direction": "NEUTRAL",
             "actualDirection": "POSITIVE",
             "relativeDirectionVerdict": "UNVERIFIED"},
        ]}},
    ]
    result = holdout.price_direction_comparison(cases)
    assert result["scoredTargets"] == 3
    assert result["modelDirectionMatches"] == 1
    assert result["modelDirectionalAccuracy"] == pytest.approx(1 / 3)
    assert result["directionalAccuracy"] == 1.0
    assert result["constantDirectionControls"]["POSITIVE"]["correct"] == 2
    assert result["constantDirectionControls"]["NEUTRAL"]["correct"] == 0
    assert result["beatsEveryConstantControl"] is False
    assert result["relativeScoredTargets"] == 2
    assert result["relativeAccuracy"] == 0.5
    assert result["attributionEstablished"] is False


def test_price_direction_empty_coverage_does_not_report_perfect_success():
    result = holdout.price_direction_comparison([
        {"evaluation": {"impacts": [{"status": "UNVERIFIED"}]}},
        {"evaluation": {"impacts": []}},
    ])
    assert result["scoredTargets"] == 0
    assert result["modelDirectionalAccuracy"] is None
    assert result["beatsEveryConstantControl"] is None


def test_content_duplicate_does_not_reenter_reserved_chronological_holdout(
    monkeypatch, tmp_path,
):
    from datetime import UTC, datetime, timedelta

    from tris_market_event_intelligence import sequential_market_learning_cli as seq

    start = datetime(2025, 1, 1, tzinfo=UTC)
    rows = []
    for i in range(10):
        t = (start + timedelta(days=i)).isoformat()
        rows.append({
            "articleId": i + 1,
            "title": f"News {i}",
            "summary": "The central bank has announced a significant "
                       "interest rate change affecting the market.",
            "publishedAtReported": t, "firstObservedAt": t,
            "textSha256": f"source-{i}",
        })
    rows[-1]["textSha256"] = rows[0]["textSha256"]
    monkeypatch.setattr(seq, "historical_news_cohort", lambda **kwargs: {
        "items": rows,
    })
    training, reserve = seq.load_articles(
        tmp_path / "news.sqlite3", 100, start + timedelta(days=20),
    )
    holdout_rows, _ = seq.load_articles(
        tmp_path / "news.sqlite3", 100, start + timedelta(days=20),
        partition="holdout",
    )
    assert reserve == 2
    assert len(training) == 8
    assert [row["articleId"] for row in holdout_rows] == [9]
    assert not ({x["textSha256"] for x in training} &
                {x["textSha256"] for x in holdout_rows})


def test_spread_date_selector_prepares_exactly_30_distinct_release_days():
    from datetime import UTC, datetime, timedelta

    from tris_market_event_intelligence.distributed_holdout_selection import (
        release_day,
        spread_date_selection,
    )

    start = datetime(2025, 1, 1, 15, tzinfo=UTC)
    rows = []
    for day in range(45):
        instant = (start + timedelta(days=day * 3)).isoformat()
        for version in range(2):
            rows.append({
                "articleId": day * 2 + version,
                "title": f"Real source fixture {day} variant {version}",
                "source": "bbc_business" if version == 0 else "bbc_technology",
                "sourceUrl": f"https://example.org/news/{day}/{version}",
                "publishedAtReported": instant,
                "firstObservedAt": instant,
                "summary": "Detailed publicly announced financial statement.",
            })
    selected, report = spread_date_selection(rows, 30)
    assert len(selected) == 30
    assert len({release_day(row) for row in selected}) == 30
    assert report["sufficientDistinctDates"] is True
    assert report["selectedSpanDays"] >= 100
    assert len(report["selectedMonths"]) >= 3
    assert report["selectionUsedFuturePrices"] is False
    assert len(report["items"]) == 30


def test_spread_date_selector_fails_closed_on_83_articles_from_three_dates():
    from datetime import UTC, datetime, timedelta

    from tris_market_event_intelligence.distributed_holdout_selection import (
        spread_date_selection,
    )

    start = datetime(2026, 11, 16, 15, tzinfo=UTC)
    rows = [{
        "articleId": i + 1,
        "title": f"Article {i}",
        "source": "bbc_world",
        "publishedAtReported": (start + timedelta(days=i % 3)).isoformat(),
        "firstObservedAt": (start + timedelta(days=i % 3)).isoformat(),
        "summary": "Substantial financial content.",
    } for i in range(83)]
    selected, report = spread_date_selection(rows, 30)
    assert len(selected) == 3
    assert report["candidateArticles"] == 83
    assert report["candidateDistinctDates"] == 3
    assert report["sufficientDistinctDates"] is False


def test_spread_date_selection_never_reuses_utc_date_across_local_market_days():
    from tris_market_event_intelligence.distributed_holdout_selection import (
        release_day,
        spread_date_selection,
    )

    rows = [
        {"articleId": 1, "source": "bbc_business", "title": "US article",
         "publishedAtReported": "2026-11-17T01:00:00+00:00"},
        {"articleId": 2, "source": "bbc_business", "title": "US article 2",
         "publishedAtReported": "2026-11-17T08:00:00+00:00"},
        {"articleId": 3, "source": "bok_all", "title": "한국은행 자료",
         "publishedAtReported": "2026-11-17T01:00:00+00:00"},
    ]
    for row in rows:
        row["firstObservedAt"] = row["publishedAtReported"]
    assert release_day(rows[0]) == "2026-11-16"
    assert release_day(rows[1]) == "2026-11-17"
    assert release_day(rows[2]) == "2026-11-17"
    selected, info = spread_date_selection(rows, 30)
    assert info["selectedDistinctDates"] == 2
    assert len(selected) == 2


def test_holdout_spread_dates_refuses_incomplete_30_day_study(monkeypatch, tmp_path):
    db = tmp_path / "news.sqlite3"
    db.touch()
    journal = tmp_path / "study.sqlite3"
    with sqlite3.connect(journal) as conn:
        conn.execute("CREATE TABLE runs(id INTEGER, settings_json TEXT)")
        conn.execute(
            "INSERT INTO runs VALUES (1, ?)",
            (json.dumps({
                "sourceDb": str(db.resolve()),
                "sourceContentCache": None,
                "cutoff": "2026-11-19T00:00:00+00:00",
                "horizonHours": 24, "interval": "1d", "model": "fake",
            }),),
        )
        conn.execute("CREATE TABLE steps(article_id INTEGER, observed_at TEXT, phase TEXT)")
        conn.execute(
            "INSERT INTO steps VALUES (1,'2026-03-01T00:00:00+00:00','COMPLETE')"
        )
    rows = [{
        "articleId": i + 5, "source": "bbc_business", "title": f"Story {i}",
        "publishedAtReported": f"2026-11-{16+i:02d}T10:00:00+00:00",
        "firstObservedAt": f"2026-11-{16+i:02d}T10:00:00+00:00",
    } for i in range(3)]
    monkeypatch.setattr(holdout, "load_articles", lambda *args, **kwargs: (rows, 10))
    selected, settings = holdout.holdout_selection(
        journal, db, 30, selection_policy="spread-dates",
    )
    assert len(selected) == 3
    assert settings["holdoutInfo"]["dateSpread"]["candidateDistinctDates"] == 3
    with pytest.raises(ValueError, match="INSUFFICIENT_DISTINCT_RELEASE_DATES"):
        holdout.evaluate_holdout(
            journal=journal, source_db=db,
            output=tmp_path / "must_not_write.json", model=object(),
            client=object(), count=30, selection_policy="spread-dates",
        )
    assert not (tmp_path / "must_not_write.json").exists()


def test_spread_dates_uses_untrained_archive_but_excludes_prior_83_holdout(
    monkeypatch, tmp_path,
):
    from datetime import UTC, datetime, timedelta

    source = tmp_path / "news.sqlite3"
    source.touch()
    journal = tmp_path / "study.sqlite3"
    with sqlite3.connect(journal) as db:
        db.execute("CREATE TABLE runs(id INTEGER, settings_json TEXT)")
        db.execute("INSERT INTO runs VALUES (1,?)", (json.dumps({
            "sourceDb": str(source.resolve()),
            "sourceContentCache": None,
            "cutoff": "2026-11-19T00:00:00+00:00",
            "horizonHours": 24, "interval": "1d", "model": "fake",
        }),))
        db.execute("CREATE TABLE steps(article_id INTEGER, observed_at TEXT, phase TEXT)")
        db.execute(
            "INSERT INTO steps VALUES (1,'2026-01-01T00:00:00+00:00','COMPLETE')"
        )
    previous = tmp_path / "market-holdout.json"
    previous.write_text(json.dumps({
        "schemaVersion": "frozen_chronological_holdout_v3",
        "holdoutSourceDb": str(source.resolve()),
        "articles": [{"articleId": 2}, {"articleId": 3}],
    }))
    start = datetime(2026, 2, 1, 15, tzinfo=UTC)
    rows = []
    for i in range(45):
        published = (start + timedelta(days=i * 4)).isoformat()
        rows.append({
            "articleId": i + 1, "source": "bbc_business",
            "title": f"Article {i+1}",
            "publishedAtReported": published, "firstObservedAt": published,
            "textSha256": f"hash-{i}",
        })
    observed_partitions = []

    def fake_loader(source_db, count, cutoff, **kwargs):
        observed_partitions.append(kwargs.get("partition"))
        return rows, 9

    monkeypatch.setattr(holdout, "load_articles", fake_loader)
    selected, settings = holdout.holdout_selection(
        journal, source, 30, selection_policy="spread-dates",
        exclude_report=previous,
    )
    assert observed_partitions == ["all"]
    assert len(selected) == 30
    assert 1 not in {row["articleId"] for row in selected}
    assert 2 not in {row["articleId"] for row in selected}
    assert 3 not in {row["articleId"] for row in selected}
    info = settings["holdoutInfo"]
    assert info["dateSpread"]["sufficientDistinctDates"]
    assert info["previouslyEvaluatedArticleIdsExcluded"] == 2
    assert info["candidateScope"] == (
        "UNTRAINED_POST_TRAINING_ARCHIVE_NOT_ORIGINAL_HOLDOUT"
    )
    assert info["dateSpread"]["selectedSpanDays"] >= 100
