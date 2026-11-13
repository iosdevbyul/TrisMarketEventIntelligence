"""Multiple claims, historical/negated statements, and durable API regression."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tris_market_event_intelligence.api import create_app
from tris_market_event_intelligence.claim_extraction import extract_claims
from tris_market_event_intelligence.claim_store import ClaimProcessor
from tris_market_event_intelligence.collector import NewsStore, parse_rss


def test_two_distinct_events_in_one_article() -> None:
    claims = extract_claims(
        "bbc_business", "Technology briefing",
        "NVIDIA CEO resigns. Company quarterly earnings released.",
    )
    assert {claim.classification for claim in claims} == {
        "CORPORATE_EVENT", "EARNINGS"
    }
    assert all(claim.status == "UNVERIFIED" for claim in claims)
    assert all(claim.claim_kind == "REPORTED" for claim in claims)


@pytest.mark.parametrize(
    ("text", "kind", "polarity", "temporal"),
    [
        ("NVIDIA CEO resigns", "REPORTED", "AFFIRMED", "UNSPECIFIED"),
        ("NVIDIA denies CEO resignation", "DENIAL", "DENIED", "UNSPECIFIED"),
        ("NVIDIA CEO reportedly resigns", "ATTRIBUTED_REPORT", "AFFIRMED", "UNSPECIFIED"),
        ("Rumored military strike", "RUMOR", "AFFIRMED", "UNSPECIFIED"),
        ("CEO might resign", "FORECAST", "AFFIRMED", "UNSPECIFIED"),
        ("NVIDIA CEO resigned last year", "HISTORICAL_REFERENCE", "AFFIRMED", "PAST"),
        ("대표이사 사임 사실무근", "DENIAL", "DENIED", "UNSPECIFIED"),
    ],
)
def test_claim_modalities(text: str, kind: str, polarity: str, temporal: str) -> None:
    result = extract_claims("bbc_business", text, None)
    assert result, text
    assert (result[0].claim_kind, result[0].polarity, result[0].temporal_context) == (
        kind, polarity, temporal
    )
    assert result[0].status != "VERIFIED"


def test_cues_are_scoped_to_individual_sentences() -> None:
    claims = extract_claims(
        "bbc_business", "Market briefing",
        "Rumored military strike. NVIDIA CEO resigns.",
    )
    by_kind = {claim.classification: claim.claim_kind for claim in claims}
    assert by_kind["GEOPOLITICAL"] == "RUMOR"
    assert by_kind["CORPORATE_EVENT"] == "REPORTED"


def test_policy_decision_only_verified_when_current_and_affirmed() -> None:
    actual = extract_claims(
        "fed_monetary", "Federal funds rate",
        "The Committee raised the interest rate by 25 basis points.",
    )
    assert any(c.status == "VERIFIED" for c in actual)
    denied = extract_claims(
        "fed_monetary", "Federal funds rate",
        "The Committee denies it raised the interest rate by 25 basis points.",
    )
    assert denied and all(c.status != "VERIFIED" for c in denied)
    historical = extract_claims(
        "fed_monetary", "Federal funds rate",
        "The Committee raised the interest rate by 25 basis points last year.",
    )
    assert historical and all(c.status != "VERIFIED" for c in historical)


def test_official_indicators_remain_candidates() -> None:
    claims = extract_claims("bls_cpi", "Consumer Price Index CPI released", None)
    assert claims[0].status == "CANDIDATE"


def test_claim_db_cursor_and_resume(tmp_path: Path) -> None:
    db = tmp_path / "claims.sqlite3"
    store = NewsStore(db)
    store.initialize()
    rss = b"""<rss><channel><item><title>Market briefing</title>
    <guid>news-1</guid><link>https://example.org/story-1</link>
    <description>NVIDIA CEO resigns. Company quarterly earnings released.</description>
    </item><item><title>Market update</title><guid>news-2</guid>
    <link>https://example.org/story-2</link>
    <description>Rumored military strike.</description></item></channel></rss>"""
    articles = parse_rss("bbc_business", rss, "2026-11-14T00:00:00+00:00")
    store.save("bbc_business", "https://example.org/feed", rss, articles, "now", None, None)
    processor = ClaimProcessor(db)
    assert processor.process_pending(batch_size=1).processed == 1
    assert processor.process_pending(batch_size=1).processed == 1
    assert processor.process_pending(batch_size=1).processed == 0
    assert len(processor.list_claims()) == 3
    with TestClient(create_app(db)) as client:
        first = client.get("/api/v1/event-claims", params={"limit": 1}).json()
        assert first["schemaVersion"] == "1.0"
        assert first["hasMore"]
        assert first["items"][0]["classification"] == "CORPORATE_EVENT"
        second = client.get(
            "/api/v1/event-claims", params={"after_id": first["nextCursor"], "limit": 1}
        ).json()
        assert second["items"][0]["classification"] == "EARNINGS"
        assert second["items"][0]["id"] > first["nextCursor"]
        assert client.get("/api/v1/event-claims", params={"limit": 0}).status_code == 422


def test_old_event_api_stays_independent(tmp_path: Path) -> None:
    db = tmp_path / "claims.sqlite3"
    assert TestClient(create_app(db)).get("/api/v1/event-claims").status_code == 503
    processor = ClaimProcessor(db)
    processor.initialize()
    with TestClient(create_app(db)) as client:
        assert client.get("/api/v1/event-claims").json()["items"] == []
        assert client.get("/api/v1/events").status_code == 200
