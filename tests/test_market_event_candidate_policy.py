"""Precommitted high-signal economic event sampling, no price-label selection."""
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from tris_market_event_intelligence.market_event_candidate_policy import (
    filter_market_event_candidates,
    research_candidate,
)
from tris_market_event_intelligence import sequential_holdout_cli as holdout


def article(
    article_id: int, title: str, summary: str,
    source: str = "bbc_business",
) -> dict:
    dt = datetime(2026, 5, 1, tzinfo=UTC) + timedelta(days=article_id)
    return {
        "articleId": article_id,
        "source": source,
        "title": title, "summary": summary,
        "firstObservedAt": dt.isoformat(),
        "publishedAtReported": dt.isoformat(),
        "sourceUrl": f"https://example.org/events/{article_id}",
        "textSha256": f"sha-{article_id}",
    }


def test_macro_indicator_and_corporate_earnings_are_eligible():
    macro = article(
        10, "2026년 3월 국제수지(잠정)",
        "2026년 3월 국제수지 발표 결과는 새로운 수출과 수입에 관한 "
        "통계의 잠정 수치를 다룬 한국은행 공식 경제지표 발표입니다.",
        "bok_statistics",
    )
    earnings = article(
        11, "Nvidia raises guidance after quarterly results",
        "Nvidia raised its quarterly revenue outlook following a reported "
        "change in demand and issued new financial guidance to investors.",
    )
    assert research_candidate(macro)["eligible"] is True
    assert research_candidate(macro)["eventCategory"] == "ECONOMIC_INDICATOR_RELEASE"
    assert research_candidate(earnings)["eligible"] is True
    assert research_candidate(earnings)["eventCategory"] == "EARNINGS_OR_GUIDANCE"


def test_media_and_irrelevant_social_stories_are_not_market_events():
    rows = [
        article(1, "Tech Life", "A weekly technology programme with the latest "
                "reviews, interviews, travel and entertainment news.", "bbc_technology"),
        article(2, "Why are more roofs not made of solar tiles?",
                "A long-form investigation into solar tile materials and "
                "the design of residential rooftops."),
        article(3, "How a civil servant got away with five jobs",
                "An investigation into a person's working arrangements in "
                "the public service and unrelated personal life."),
    ]
    kept, report = filter_market_event_candidates(rows)
    assert kept == []
    assert report["economicEventCandidates"] == 0
    assert report["rejectedCandidateArticles"] == 3
    assert report["selectionUsesRealizedMarketReturns"] is False


def test_corrections_and_title_only_macro_bodies_do_not_pass():
    correction = article(
        4, "한국은행 통화정책 관련 보도해명자료",
        "보도내용 한국은행이 새로운 금리 결정을 발표했다고 전했습니다. "
        "해명내용 보도 내용은 사실과 다름으로 해당 결정은 없었습니다.",
        "bok_corrections",
    )
    title_only = article(5, "2026년 4월 국제수지(잠정)",
                         "자세한 내용은 첨부파일을 참고하시기 바랍니다.",
                         "bok_statistics")
    assert not research_candidate(correction)["eligible"]
    assert "OFFICIAL_CORRECTION_NOT_NEW_CONFIRMED_ACTION" in (
        research_candidate(correction)["reasons"]
    )
    assert "INSUFFICIENT_OR_PLACEHOLDER_BODY" in (
        research_candidate(title_only)["reasons"]
    )


def test_30_distinct_event_dates_requires_real_qualifying_content():
    from tris_market_event_intelligence.distributed_holdout_selection import (
        release_day,
        spread_date_selection,
    )

    rows = []
    for i in range(65):
        title = "2026년 3월 국제수지(잠정)" if i % 2 == 0 else "Tech Now"
        summary = (
            "국제수지 공식 통계에 따른 수출 및 수입 발표의 자세한 "
            "경제지표 분석 수치와 새로운 시장 동향을 설명합니다."
            if i % 2 == 0 else
            "A weekly television programme about new apps and technology."
        )
        rows.append(article(i + 1, title, summary,
                            "bok_statistics" if i % 2 == 0 else "bbc_technology"))
    filtered, audit = filter_market_event_candidates(rows)
    selected, dates = spread_date_selection(filtered, 30)
    assert len(selected) == 30
    assert len({release_day(row) for row in selected}) == 30
    assert dates["sufficientDistinctDates"]
    assert audit["economicEventCandidates"] == 33
    assert audit["rejectedCandidateArticles"] == 32


def test_historical_events_require_preflight_and_unmodified_manifest(
    monkeypatch, tmp_path,
):
    source = tmp_path / "news.sqlite3"
    source.touch()
    journal = tmp_path / "train.sqlite3"
    with sqlite3.connect(journal) as db:
        db.execute("CREATE TABLE runs(id INTEGER, settings_json TEXT)")
        db.execute("INSERT INTO runs VALUES (1,?)", (json.dumps({
            "sourceDb": str(source.resolve()),
            "sourceContentCache": None,
            "cutoff": "2026-11-19T00:00:00+00:00",
            "horizonHours": 24, "interval": "1d", "model": "fake",
        }),))
        db.execute(
            "CREATE TABLE steps(article_id INTEGER, observed_at TEXT, phase TEXT)"
        )
        db.execute(
            "INSERT INTO steps VALUES (1,'2026-04-01T00:00:00+00:00','COMPLETE')"
        )
    rows = [
        article(i + 2, "2026년 3월 국제수지(잠정)",
                "국제수지 공식 통계의 신규 발표이며 수출입 수지와 "
                "대외 경제지표 결과를 설명하는 자료입니다.", "bok_statistics")
        for i in range(36)
    ]
    rows += [
        article(i + 100, "Tech Life",
                "A weekly talk programme featuring unrelated personal stories.",
                "bbc_technology")
        for i in range(10)
    ]
    previous_one = tmp_path / "first.json"
    previous_two = tmp_path / "second.json"
    for path, articles in ((previous_one, [2, 3]), (previous_two, [4])):
        path.write_text(json.dumps({
            "schemaVersion": "frozen_chronological_holdout_v3",
            "holdoutSourceDb": str(source.resolve()),
            "articles": [{"articleId": i} for i in articles],
        }))
    monkeypatch.setattr(holdout, "load_articles",
                        lambda *args, **kwargs: (rows, 10))
    selected, settings = holdout.holdout_selection(
        journal, source, 30, selection_policy="event-spread-dates",
        exclude_report=[previous_one, previous_two],
    )
    assert len(selected) == 30
    assert {2, 3, 4}.isdisjoint({row["articleId"] for row in selected})
    info = settings["holdoutInfo"]
    assert info["economicEventScreen"]["economicEventCandidates"] == 33
    assert info["economicEventScreen"]["rejectedCandidateArticles"] == 10
    assert len(info["dateSpread"]["items"]) == 30
    assert all(i["marketEventEligibility"]["eligible"]
               for i in info["dateSpread"]["items"])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(info))
    # Avoid invoking the model: a mismatched frozen manifest should be
    # rejected before the first price or Gemma request.
    report = tmp_path / "output.json"
    monkeypatch.setattr(holdout, "fit_price_calibrator",
                        lambda *a, **kw: {
                            "status": "NO_SUPPORTED_RULES", "rules": {},
                            "ruleCount": 0,
                        })
    with pytest.raises(ValueError, match="FROZEN_ARTICLE_MANIFEST_MISMATCH"):
        changed = dict(info)
        changed["selectionPolicy"] = "chronological"
        manifest.write_text(json.dumps(changed))
        holdout.evaluate_holdout(
            journal=journal, source_db=source, output=report,
            model=object(), client=object(), count=30,
            selection_policy="event-spread-dates",
            exclude_report=[previous_one, previous_two],
            selection_manifest=manifest,
        )
    assert not report.exists()
