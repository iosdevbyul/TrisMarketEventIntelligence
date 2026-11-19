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
            "CREATE TABLE steps(article_id INTEGER, observed_at TEXT, phase TEXT)"
        )
        db.execute(
            "INSERT INTO steps VALUES (1,'2024-01-01T00:00:00+00:00','COMPLETE')"
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
