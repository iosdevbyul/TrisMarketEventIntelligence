"""Claim assessments distinguish review records from verified facts."""

from pathlib import Path

from fastapi.testclient import TestClient

from tris_market_event_intelligence.api import create_app
from tris_market_event_intelligence.claim_assessment import assess_claim
from tris_market_event_intelligence.claim_review import ClaimReviewManager
from tris_market_event_intelligence.claim_store import ClaimProcessor
from tris_market_event_intelligence.collector import NewsStore, parse_rss


def test_rule_status_is_not_fact_confirmation() -> None:
    row = assess_claim(
        claim_id=1, classification="MONETARY_POLICY", status="VERIFIED",
        claim_kind="REPORTED", polarity="AFFIRMED",
        temporal_context="UNSPECIFIED", review_status=None, revision=0,
    )
    assert row["fact_confirmed"] is False
    assert row["verification_level"] == "RULE_SUPPORTED_NOT_INDEPENDENTLY_VERIFIED"
    assert row["review_priority"] == "REVIEW_FIRST"


def test_nonreporting_and_historical_claims_are_deferred() -> None:
    for kind in ("RUMOR", "FORECAST", "DENIAL", "HISTORICAL_REFERENCE"):
        row = assess_claim(
            claim_id=1, classification="ECONOMIC_INDICATOR",
            status="UNVERIFIED", claim_kind=kind, polarity="AFFIRMED",
            temporal_context="UNSPECIFIED", review_status=None, revision=0,
        )
        assert row["analysis_suitability"] == "EXCLUDE_PENDING_REVIEW"
        assert row["review_priority"] == "DEFER"
    old = assess_claim(
        claim_id=1, classification="CORPORATE_EVENT", status="UNVERIFIED",
        claim_kind="REPORTED", polarity="AFFIRMED",
        temporal_context="PAST", review_status=None, revision=0,
    )
    assert old["analysis_suitability"] == "EXCLUDE_PENDING_REVIEW"


def test_manual_review_states_do_not_certify_truth() -> None:
    cases = [
        ("DISPUTED", "EXCLUDE_PENDING_REVIEW", "REVIEW_FIRST"),
        ("RETRACTED", "EXCLUDE_PENDING_REVIEW", "REVIEW_FIRST"),
        ("CORROBORATED", "CONDITIONAL_RESEARCH", "REVIEW_FIRST"),
        ("CONFIRMED", "HUMAN_REVIEWED_RESEARCH", "NORMAL"),
    ]
    for status, suitability, priority in cases:
        row = assess_claim(
            claim_id=1, classification="CORPORATE_EVENT", status="UNVERIFIED",
            claim_kind="REPORTED", polarity="AFFIRMED",
            temporal_context="UNSPECIFIED", review_status=status, revision=2,
        )
        assert row["analysis_suitability"] == suitability
        assert row["review_priority"] == priority
        assert row["fact_confirmed"] is False


def _fixture(path: Path) -> tuple[int, int]:
    store = NewsStore(path)
    store.initialize()
    payload = b"""<rss><channel>
    <item><guid>1</guid><title>NVIDIA CEO resigns</title>
    <link>https://example.org/a</link></item>
    <item><guid>2</guid><title>NVIDIA denies CEO resignation</title>
    <link>https://example.org/b</link></item>
    </channel></rss>"""
    store.save(
        "bbc_business", "https://example.org/rss", payload,
        parse_rss("bbc_business", payload, "now"), "now", None, None,
    )
    ClaimProcessor(path).process_pending()
    with store.connect() as db:
        claim = db.execute(
            "SELECT id FROM article_event_claims ORDER BY id LIMIT 1"
        ).fetchone()[0]
        denial_article = db.execute(
            "SELECT id FROM articles ORDER BY id LIMIT 1 OFFSET 1"
        ).fetchone()[0]
    return claim, denial_article


def test_readonly_api_and_review_revision(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    claim, denial_article = _fixture(path)
    client = TestClient(create_app(path))
    first = client.get("/api/v1/claim-assessments", params={"limit": 1})
    assert first.status_code == 200
    row = first.json()["items"][0]
    assert row["claim_id"] == claim
    assert row["latest_review_status"] is None
    assert first.json()["hasMore"] is True
    next_page = client.get("/api/v1/claim-assessments", params={
        "after_id": first.json()["nextCursor"], "limit": 2,
    }).json()
    assert next_page["items"]
    assert all(item["claim_id"] > claim for item in next_page["items"])
    assert client.get("/api/v1/claim-assessments", params={"limit": 0}).status_code == 422
    assert client.post("/api/v1/claim-assessments", json={}).status_code == 405

    ClaimReviewManager(path).record(
        claim_id=claim, status="DISPUTED",
        evidence_article_id=denial_article,
        evidence_quote="NVIDIA denies CEO resignation",
        evidence_kind="DENIAL",
        reason="Published denial requires manual review",
        expected_revision=0,
    )
    after = client.get("/api/v1/claim-assessments", params={"limit": 1}).json()
    assert after["nextCursor"] == first.json()["nextCursor"]
    assert after["items"][0]["review_revision"] == 1
    assert after["items"][0]["latest_review_status"] == "DISPUTED"
    assert after["items"][0]["analysis_suitability"] == "EXCLUDE_PENDING_REVIEW"
    with NewsStore(path).connect() as conn:
        assert conn.execute(
            "SELECT status FROM article_event_claims WHERE id=?", (claim,)
        ).fetchone()[0] == "UNVERIFIED"


def test_missing_db_returns_503_without_creation(tmp_path: Path) -> None:
    missing = tmp_path / "absent.sqlite3"
    client = TestClient(create_app(missing))
    assert client.get("/api/v1/claim-assessments").status_code == 503
    assert not missing.exists()
