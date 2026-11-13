"""Multi-category extraction regression tests; all news content is fictional."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tris_market_event_intelligence.api import create_app
from tris_market_event_intelligence.collector import NewsStore, parse_rss
from tris_market_event_intelligence.event_store import EventProcessor
from tris_market_event_intelligence.multi_events import extract_news_event


@pytest.mark.parametrize(
    ("source", "title", "category", "status", "claim"),
    [
        ("bbc_business", "NVIDIA CEO resigns", "CORPORATE_EVENT", "UNVERIFIED", "REPORTED"),
        ("bbc_business", "Quarterly earnings announced", "EARNINGS", "UNVERIFIED", "REPORTED"),
        ("bls_cpi", "Consumer Price Index CPI released", "ECONOMIC_INDICATOR", "CANDIDATE", "REPORTED"),
        ("bbc_world", "Military strike hits region", "GEOPOLITICAL", "UNVERIFIED", "REPORTED"),
        ("bbc_business", "Tariffs on semiconductor exports", "REGULATION", "UNVERIFIED", "REPORTED"),
        ("bbc_technology", "GPU supply shortage deepens", "SUPPLY_CHAIN", "UNVERIFIED", "REPORTED"),
        ("bbc_technology", "Cloud service outage disrupts users", "SERVICE_INCIDENT", "UNVERIFIED", "REPORTED"),
        ("bbc_business", "KOSPI plunges on trading day", "MARKET_MOVEMENT", "UNVERIFIED", "REPORTED"),
        ("bbc_business", "CEO reportedly resigns", "CORPORATE_EVENT", "UNVERIFIED", "ATTRIBUTED_REPORT"),
        ("bbc_world", "Rumored missile strike", "GEOPOLITICAL", "UNVERIFIED", "RUMOR"),
        ("bbc_business", "대표이사 사임", "CORPORATE_EVENT", "UNVERIFIED", "REPORTED"),
        ("bbc_business", "반도체 공급 부족", "SUPPLY_CHAIN", "UNVERIFIED", "REPORTED"),
    ],
)
def test_category(source: str, title: str, category: str, status: str, claim: str) -> None:
    result = extract_news_event(source, title, None)
    assert result is not None
    assert (result.event.classification, result.event.status, result.claim_kind) == (
        category, status, claim
    )
    assert result.event.extractor_version == "multi-v2"


def test_official_policy_is_only_rule_verified_category() -> None:
    result = extract_news_event(
        "fed_monetary", "Federal funds target rate",
        "The Committee raised the interest rate by 25 basis points."
    )
    assert result is not None
    assert (result.event.status, result.event.change_basis_points) == ("VERIFIED", 25)
    forecast = extract_news_event(
        "fed_monetary", "Federal funds rate",
        "The Committee is expected to raise the interest rate by 25 basis points."
    )
    assert forecast is not None
    assert forecast.event.status == "CANDIDATE"
    assert forecast.event.change_basis_points is None


def test_untrusted_policy_claim_is_never_verified() -> None:
    result = extract_news_event(
        "bbc_business", "Federal funds rate",
        "The Committee raised the interest rate by 25 basis points."
    )
    assert result is not None
    assert result.event.status == "UNVERIFIED"


def test_story_without_specific_trigger_is_ignored() -> None:
    assert extract_news_event("bbc_business", "Weather this weekend", None) is None


def test_multiple_tags() -> None:
    result = extract_news_event(
        "bbc_technology", "Export controls and semiconductor supply shortage", None
    )
    assert result is not None
    assert "TRADE_POLICY" in result.tags
    assert "SEMICONDUCTOR" in result.tags


def test_persistence_and_api_metadata(tmp_path: Path) -> None:
    path = tmp_path / "event.sqlite3"
    store = NewsStore(path)
    store.initialize()
    rss = b"""<rss><channel><item><title>NVIDIA CEO resigns</title>
    <link>https://example.org/1</link><guid>1</guid></item>
    <item><title>Consumer Price Index released</title>
    <link>https://example.org/2</link><guid>2</guid></item></channel></rss>"""
    for source in ("bbc_business", "bls_cpi"):
        articles = parse_rss(source, rss, "2026-11-14T00:00:00+00:00")
        store.save(source, "https://example.org/feed", rss, articles, "now", None, None)
    processor = EventProcessor(path)
    stats = processor.process_pending()
    assert stats.processed == 4
    assert stats.events == 4
    assert processor.process_pending().processed == 0
    with store.connect() as db:
        metadata = db.execute("SELECT claim_kind, tags_json FROM event_metadata").fetchall()
        assert len(metadata) == 4
        assert all(json.loads(row[1]) for row in metadata)
    # The impact schema is initialized for readiness without making model requests.
    from tris_market_event_intelligence.impact_store import ImpactProcessor

    class NoCallProvider:
        model_id = "no-call"

        def analyze(self, event):
            raise AssertionError("Should not invoke a model")

    ImpactProcessor(path, NoCallProvider()).initialize()
    response = TestClient(create_app(path)).get("/api/v1/events")
    assert response.status_code == 200
    items = response.json()["items"]
    assert items[0]["claimKind"] == "REPORTED"
    assert items[0]["tags"] == ["LEADERSHIP"]
    assert items[0]["verificationNote"] == "THIRD_PARTY_REPORT"
