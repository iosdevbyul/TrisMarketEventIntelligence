"""Regression checks for persisted per-article AI prediction evidence audit."""
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence import sequential_article_audit_cli as audit


def build(path: Path):
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE runs (id INTEGER PRIMARY KEY, settings_json TEXT);
            CREATE TABLE steps (
                seq INTEGER PRIMARY KEY, article_id INTEGER, observed_at TEXT,
                prediction_json TEXT, evaluation_json TEXT, feedback_json TEXT,
                phase TEXT
            );
        """)
        db.execute(
            "INSERT INTO runs VALUES (1,?)",
            (json.dumps({"cutoff": "2026-11-19T00:00:00+09:00"}),),
        )
        db.execute(
            "INSERT INTO steps VALUES (?,?,?,?,?,?,?)",
            (
                1, 7, "2026-11-15T10:00:00+00:00",
                json.dumps({"summary": "Bank margin pressure", "impacts": [{
                    "targetType": "SECTOR", "target": "FINANCIALS",
                    "priceProxy": "XLF", "direction": "NEUTRAL",
                    "reason": "Opposing interest-rate effects",
                }]}),
                json.dumps({"source": "Yahoo Finance chart",
                            "interval": "1d", "horizonHours": 24,
                            "impacts": [{
                                "targetType": "SECTOR", "target": "FINANCIALS",
                                "priceProxy": "XLF", "direction": "NEUTRAL",
                                "actualDirection": "NEGATIVE", "status": "WRONG",
                                "absoluteReturn": -0.02,
                                "baselinePrice": 100,
                                "outcomePrice": 98,
                                "benchmarkReturn": -0.01,
                                "excessReturn": -0.01,
                                "baselineAt": "2026-11-14T14:30:00+00:00",
                                "outcomeAt": "2026-11-17T14:30:00+00:00",
                            }]}),
                json.dumps({"diagnosis": "Ignored expectations",
                            "lesson": "Check rate expectations before direction"}),
                "COMPLETE",
            ),
        )
        db.commit()


def test_audit_shows_prediction_outcome_and_feedback(tmp_path):
    path = tmp_path / "journal.sqlite3"
    build(path)
    result = audit.inspect(path)
    assert result["summary"]["predictionsCorrect"] == 0
    assert result["summary"]["predictionsWrong"] == 1
    assert result["summary"]["neutralPredictionsMarkedWrong"] == 1
    row = result["articles"][0]
    assert row["outcomes"][0]["predictedDirection"] == "NEUTRAL"
    assert row["outcomes"][0]["observedDirection"] == "NEGATIVE"
    assert row["outcomes"][0]["absoluteReturn"] == -0.02
    assert row["outcomes"][0]["baselinePrice"] == 100
    assert row["outcomes"][0]["outcomePrice"] == 98
    assert row["outcomes"][0]["predictionReason"] == "Opposing interest-rate effects"
    assert row["correctiveRule"] == "Check rate expectations before direction"
    md = audit.markdown(result)
    assert "금융" not in md or "FINANCIALS" in md
    assert "FINANCIALS" in md
    assert "-2.000%" in md
    assert "Check rate expectations" in md
    page = audit.html_report(result)
    assert "FINANCIALS" in page
    assert "NEUTRAL" in page
    assert "NEGATIVE" in page
    assert "100" in page and "98" in page
    assert "Bank margin pressure" in page


def test_source_metadata_loaded_and_report_is_read_only(monkeypatch, tmp_path):
    path = tmp_path / "journal.sqlite3"
    source = tmp_path / "news.sqlite3"
    source.touch()
    build(path)

    def fake_cohort(**kwargs):
        assert kwargs["after_id"] == 0
        assert kwargs["as_of"] == datetime(
            2026, 11, 18, 15, 0, tzinfo=UTC,
        )
        return {"items": [{
            "articleId": 7, "title": "Central bank news",
            "summary": "Inflation and new rules", "source": "RSS",
            "sourceUrl": "https://example.org/a",
            "firstObservedAt": "2026-11-15T10:00:00+00:00",
        }]}

    monkeypatch.setattr(audit, "historical_news_cohort", fake_cohort)
    original = path.read_bytes()
    result = audit.inspect(path, source_db=source)
    assert result["articles"][0]["title"] == "Central bank news"
    assert "Central bank news" in audit.markdown(result)
    assert path.read_bytes() == original


def test_no_predictions_and_unverified_not_counted_as_hits(tmp_path):
    path = tmp_path / "journal.sqlite3"
    build(path)
    with sqlite3.connect(path) as db:
        db.execute(
            "INSERT INTO steps VALUES (?,?,?,?,?,?,?)",
            (2, 8, "2026-11-15T11:00:00+00:00",
             '{"impacts":[]}', '{"impacts":[]}', '{"lesson":"","diagnosis":""}',
             "COMPLETE"),
        )
        db.commit()
    report = audit.inspect(path)
    assert report["summary"]["articlesShown"] == 2
    assert "NO_TARGETS" in audit.markdown(report)


def test_html_groups_targets_and_links_and_retains_unevaluated(tmp_path, monkeypatch):
    path = tmp_path / "journal.sqlite3"
    source = tmp_path / "news.sqlite3"
    source.touch()
    build(path)
    with sqlite3.connect(path) as db:
        prediction = {
            "summary": "Mixed effects",
            "impacts": [
                {"targetType": "SECTOR", "target": "FINANCIALS",
                 "priceProxy": "XLF", "direction": "NEUTRAL",
                 "reason": "Sector impact"},
                {"targetType": "STOCK", "target": "JPM",
                 "priceProxy": "JPM", "direction": "POSITIVE",
                 "reason": "Bank impact"},
            ],
        }
        db.execute("UPDATE steps SET prediction_json=? WHERE seq=1",
                   (json.dumps(prediction),))
        db.commit()

    monkeypatch.setattr(audit, "historical_news_cohort", lambda **kwargs: {
        "items": [{"articleId": 7, "title": "Article <one>",
                   "summary": "News summary", "source": "RSS",
                   "sourceUrl": "https://example.org/article?a=1&b=2"}],
    })
    report = audit.inspect(path, source_db=source)
    rows = report["articles"][0]["outcomes"]
    assert len(rows) == 2
    assert any(row["target"] == "JPM" and row["verdict"] == "UNVERIFIED"
               for row in rows)
    page = audit.html_report(report)
    assert "연결된 섹터 (1)" in page
    assert "연결된 종목 (1)" in page
    assert "시장·지수 (0)" in page
    assert "상승" in page and "중립" in page
    assert "NO_STORED_EVALUATION" in page
    assert "https://example.org/article?a=1&amp;b=2" in page
    assert "Article &lt;one&gt;" in page


def test_html_source_url_rejects_javascript_scheme():
    report = {
        "summary": {"articlesShown": 1, "predictionsCorrect": 0,
                    "predictionsWrong": 0, "predictionsUnverified": 0},
        "warnings": [],
        "articles": [{
            "seq": 1, "articleId": 1, "title": "Unsafe URL",
            "firstObservedAt": "2026-11-15T00:00:00+00:00",
            "source": "Example", "sourceUrl": "javascript:alert(1)",
            "articleSummary": "Summary", "originalPredictionSummary": "Neutral",
            "outcomes": [], "diagnosis": "", "correctiveRule": "",
            "rulesApplied": [],
        }],
    }
    html = audit.html_report(report)
    assert "원문 링크 없음" in html
    assert "href=" not in html


def test_audit_displays_rejected_mapping_without_fabricating_outcome(tmp_path):
    path = tmp_path / "journal.sqlite3"
    build(path)
    with sqlite3.connect(path) as db:
        row = db.execute("SELECT prediction_json FROM steps WHERE seq=1").fetchone()
        prediction = json.loads(row[0])
        prediction["targetMappingRejections"] = [{
            "targetType": "SECTOR",
            "target": "UNLISTED_INDUSTRY",
            "reasonCode": "UNSUPPORTED_NON_STOCK_TARGET",
        }]
        prediction["recoveryStatus"] = "PARTIAL_TARGET_MAPPING"
        db.execute("UPDATE steps SET prediction_json=? WHERE seq=1",
                   (json.dumps(prediction),))
        db.commit()
    report = audit.inspect(path)
    page = audit.html_report(report)
    assert "UNLISTED_INDUSTRY" in page
    assert "PARTIAL_TARGET_MAPPING" in page
    assert report["summary"]["predictionsWrong"] == 1
