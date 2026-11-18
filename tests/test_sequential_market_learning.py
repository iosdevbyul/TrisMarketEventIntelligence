"""Regression tests for sequential, resumable market-learning loop."""
from __future__ import annotations

import json
import sqlite3
from argparse import Namespace
from datetime import UTC, datetime, timedelta

import pytest

from tris_market_event_intelligence import sequential_market_learning_cli as seq


def test_prediction_target_validation():
    prediction = seq.canonical_prediction({
        "summary": "An event",
        "impacts": [
            {"targetType": "STOCK", "target": "NVDA", "direction": "NEGATIVE",
             "reason": "Supply restrictions"},
            {"targetType": "SECTOR", "target": "SEMICONDUCTORS",
             "direction": "NEGATIVE", "reason": "Demand"},
        ],
    })
    assert prediction["impacts"][1]["priceProxy"] == "SMH"
    with pytest.raises(ValueError, match="Unsupported"):
        seq.canonical_prediction({"impacts": [{
            "targetType": "SECTOR", "target": "NOT_A_SECTOR",
            "direction": "POSITIVE", "reason": "Unknown",
        }]})


class ScriptedModel:
    def __init__(self):
        self.prompts = []
        self.predictions = 0

    def generate_json(self, *, system, user, schema):
        payload = json.loads(user)
        self.prompts.append(payload)
        if "wrongOutcomes" in payload:
            return json.dumps({
                "diagnosis": "The negative scenario overlooked strong demand",
                "lesson": "When production is constrained, check whether demand "
                          "still supports pricing power before predicting a decline.",
            })
        self.predictions += 1
        return json.dumps({
            "summary": "Supply-side headline",
            "impacts": [
                {"targetType": "STOCK", "target": "NVDA",
                 "direction": "NEGATIVE" if self.predictions == 1 else "POSITIVE",
                 "reason": "Supply pressures"},
            ],
        })


def sample(article_id, date):
    return {
        "articleId": article_id, "source": "test",
        "title": "Chip manufacturing supply event",
        "summary": "A reported production shift could influence supplier revenue",
        "firstObservedAt": date.isoformat(),
    }


def args(tmp_path):
    return Namespace(
        source_db=tmp_path / "news.sqlite3",
        journal=tmp_path / "learning.sqlite3",
        as_of="2026-11-19T00:00:00+00:00",
        count=2, horizon_hours=24, interval="1d", model="gemma4:e2b",
    )


def test_wrong_answer_updates_next_article_and_resume(monkeypatch, tmp_path):
    first = datetime(2025, 1, 1, tzinfo=UTC)
    articles = [sample(10, first), sample(11, first + timedelta(days=7))]
    monkeypatch.setattr(seq, "load_articles",
                        lambda db, count, cutoff: (articles, 1))

    def fake_price(client, symbol, observed, horizon, interval):
        assert symbol in {"NVDA", "SPY"}
        return {
            "return": 0.03 if symbol == "NVDA" else 0.01,
            "baselineAt": (observed - timedelta(hours=1)).isoformat(),
            "outcomeAt": (observed + timedelta(days=2)).isoformat(),
        }

    monkeypatch.setattr(seq, "price_return", fake_price)
    fake = ScriptedModel()
    config = args(tmp_path)
    result = seq.run(config, model=fake, client=object())
    assert result["articlesCompleted"] == 2
    assert result["lessons"] == 1
    assert fake.predictions == 2
    assert fake.prompts[0]["priorFeedbackLessons"] == []
    assert "pricing power" in fake.prompts[2]["priorFeedbackLessons"][0]
    with sqlite3.connect(config.journal) as db:
        predictions = db.execute(
            "SELECT prediction_json, phase FROM steps ORDER BY seq"
        ).fetchall()
        assert json.loads(predictions[0][0])["impacts"][0]["direction"] == "NEGATIVE"
        assert json.loads(predictions[1][0])["impacts"][0]["direction"] == "POSITIVE"
        assert [r[1] for r in predictions] == ["COMPLETE", "COMPLETE"]
    resumed = ScriptedModel()
    seq.run(config, model=resumed, client=object())
    assert resumed.prompts == []


def test_no_feedback_on_unavailable_price(monkeypatch, tmp_path):
    articles = [sample(1, datetime(2025, 1, 1, tzinfo=UTC))]
    monkeypatch.setattr(seq, "load_articles",
                        lambda db, count, cutoff: (articles, 0))
    def missing(*args, **kwargs):
        raise ValueError("No historical bars")

    monkeypatch.setattr(seq, "price_return", missing)
    config = args(tmp_path)
    config.count = 1
    fake = ScriptedModel()
    result = seq.run(config, model=fake, client=object())
    assert result["marketReturnsAnalyzed"] is False
    assert result["lessons"] == 0
    assert len(fake.prompts) == 1


def test_learning_rejects_overlapping_price_outcomes(monkeypatch, tmp_path):
    first = datetime(2025, 1, 1, tzinfo=UTC)
    articles = [sample(10, first), sample(11, first + timedelta(days=6))]
    monkeypatch.setattr(seq, "load_articles",
                        lambda db, count, cutoff: (articles, 0))
    def future_price(client, symbol, observed, horizon, interval):
        return {
            "return": 0.05,
            "baselineAt": (observed - timedelta(days=1)).isoformat(),
            "outcomeAt": (observed + timedelta(days=7)).isoformat(),
        }

    monkeypatch.setattr(seq, "price_return", future_price)
    with pytest.raises(ValueError, match="outcome overlaps"):
        seq.run(args(tmp_path), model=ScriptedModel(), client=object())
