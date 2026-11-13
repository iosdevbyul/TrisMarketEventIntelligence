"""Human review, evidence provenance and immutable revision history."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tris_market_event_intelligence.api import create_app
from tris_market_event_intelligence.claim_review import (
    ClaimReviewManager,
    ReviewConflict,
)
from tris_market_event_intelligence.claim_store import ClaimProcessor
from tris_market_event_intelligence.collector import NewsStore, parse_rss


def _fixture(path: Path) -> tuple[int, int, int]:
    news = NewsStore(path)
    news.initialize()
    ordinary = b"""<rss><channel><item><title>NVIDIA CEO resigns</title>
    <guid>1</guid><link>https://example.org/news-1</link>
    </item><item><title>NVIDIA denies CEO resignation</title>
    <guid>2</guid><link>https://example.org/news-2</link>
    </item></channel></rss>"""
    articles = parse_rss("bbc_business", ordinary, "2026-11-14T00:00:00+00:00")
    news.save("bbc_business", "https://example.org/rss", ordinary, articles,
              "now", None, None)
    official = b"""<rss><channel><item><title>Federal funds rate</title>
    <guid>fed</guid><link>https://www.federalreserve.gov/monetarypolicy/fomc.htm</link>
    <description>The Committee raised the interest rate by 25 basis points.</description>
    </item></channel></rss>"""
    rows = parse_rss("fed_monetary", official, "2026-11-14T00:00:00+00:00")
    news.save("fed_monetary", "https://www.federalreserve.gov/feeds/press_monetary.xml",
              official, rows, "now", None, None)
    ClaimProcessor(path).process_pending()
    with news.connect() as db:
        claim_id = db.execute(
            "SELECT id FROM article_event_claims WHERE classification='CORPORATE_EVENT' "
            "ORDER BY id LIMIT 1"
        ).fetchone()[0]
        articles = db.execute("SELECT id FROM articles ORDER BY id").fetchall()
    return claim_id, articles[1][0], articles[2][0]


def test_multiple_reviews_preserve_full_history_and_source_claim(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "review.sqlite3"
    claim, denial_article, official_article = _fixture(db_path)
    manager = ClaimReviewManager(db_path)
    first = manager.record(
        claim_id=claim, status="UNDER_REVIEW",
        evidence_article_id=denial_article, evidence_quote="NVIDIA denies CEO resignation",
        evidence_kind="DENIAL", reason="Checking whether news report is reliable",
        expected_revision=0, operation_id="first",
    )
    assert first.prior_status == "UNVERIFIED"
    second = manager.record(
        claim_id=claim, status="DISPUTED",
        evidence_article_id=denial_article, evidence_quote="NVIDIA denies CEO resignation",
        evidence_kind="DENIAL", reason="Contradictory denial has been published",
        expected_revision=1, operation_id="second",
    )
    assert second.prior_status == "UNDER_REVIEW"
    third = manager.record(
        claim_id=claim, status="UNDER_REVIEW",
        evidence_article_id=official_article, evidence_quote="Federal funds rate",
        evidence_kind="SUPPORTING_REPORT", reason="Opening review for additional confirmation",
        expected_revision=2,
    )
    assert third.prior_status == "DISPUTED"
    with manager.claims.store.connect() as conn:
        original = conn.execute(
            "SELECT status FROM article_event_claims WHERE id=?", (claim,)
        ).fetchone()[0]
        events = conn.execute(
            "SELECT revision, prior_status, status FROM claim_review_history "
            "WHERE claim_id=? ORDER BY revision", (claim,),
        ).fetchall()
    assert original == "UNVERIFIED"
    assert events == [
        (1, "UNVERIFIED", "UNDER_REVIEW"),
        (2, "UNDER_REVIEW", "DISPUTED"),
        (3, "DISPUTED", "UNDER_REVIEW"),
    ]


def test_confirmation_requires_official_primary_evidence(tmp_path: Path) -> None:
    path = tmp_path / "review.sqlite3"
    claim, denial_article, official_article = _fixture(path)
    manager = ClaimReviewManager(path)
    common = dict(
        claim_id=claim, status="CONFIRMED", evidence_kind="PRIMARY_DOCUMENT",
        reason="I personally reviewed this original statement", expected_revision=0,
    )
    with pytest.raises(ValueError, match="official"):
        manager.record(
            **common, evidence_article_id=denial_article,
            evidence_quote="NVIDIA denies CEO resignation",
        )
    # Even an authentic monetary-policy release cannot prove an unrelated
    # claim about the CEO of a company.
    with pytest.raises(ValueError, match="does not corroborate"):
        manager.record(
            **common, evidence_article_id=official_article,
            evidence_quote="The Committee raised the interest rate by 25 basis points.",
        )
    with manager.claims.store.connect() as conn:
        policy_claim = conn.execute(
            "SELECT id FROM article_event_claims "
            "WHERE classification='MONETARY_POLICY' ORDER BY id LIMIT 1"
        ).fetchone()[0]
    confirmed = manager.record(
        **{**common, "claim_id": policy_claim},
        evidence_article_id=official_article,
        evidence_quote="The Committee raised the interest rate by 25 basis points.",
    )
    assert confirmed.status == "CONFIRMED"


def test_evidence_quote_and_revision_conflicts(tmp_path: Path) -> None:
    path = tmp_path / "review.sqlite3"
    claim, denial_article, _ = _fixture(path)
    manager = ClaimReviewManager(path)
    kwargs = dict(
        claim_id=claim, status="DISPUTED", evidence_article_id=denial_article,
        evidence_quote="NVIDIA denies CEO resignation",
        evidence_kind="DENIAL", reason="News publisher has published a denial",
        expected_revision=0, operation_id="retry-key",
    )
    with pytest.raises(ValueError, match="verbatim"):
        manager.record(**{**kwargs, "evidence_quote": "invented citation"})
    first = manager.record(**kwargs)
    assert manager.record(**kwargs).revision_id == first.revision_id
    with pytest.raises(ReviewConflict, match="Stale revision"):
        manager.record(**{**kwargs, "operation_id": "later"})
    with pytest.raises(ValueError, match="already used"):
        manager.record(**{**kwargs, "status": "UNDER_REVIEW"})


def test_retraction_requires_correction_evidence(tmp_path: Path) -> None:
    path = tmp_path / "review.sqlite3"
    claim, denial_article, _ = _fixture(path)
    manager = ClaimReviewManager(path)
    with pytest.raises(ValueError, match="correction"):
        manager.record(
            claim_id=claim, status="RETRACTED", evidence_article_id=denial_article,
            evidence_quote="NVIDIA denies CEO resignation",
            evidence_kind="DENIAL", reason="Claim retracted by the source",
            expected_revision=0,
        )


def test_review_api_independent_cursors_and_read_only(tmp_path: Path) -> None:
    path = tmp_path / "review.sqlite3"
    claim, denial_article, _ = _fixture(path)
    manager = ClaimReviewManager(path)
    for revision, status in enumerate(("UNDER_REVIEW", "DISPUTED")):
        manager.record(
            claim_id=claim, status=status, evidence_article_id=denial_article,
            evidence_quote="NVIDIA denies CEO resignation", evidence_kind="DENIAL",
            reason="Evidence considered by the local reviewer", expected_revision=revision,
        )
    client = TestClient(create_app(path))
    page = client.get("/api/v1/claim-review-history", params={"limit": 1}).json()
    assert page["hasMore"] and page["items"][0]["revision"] == 1
    next_page = client.get("/api/v1/claim-review-history", params={
        "limit": 1, "after_id": page["nextCursor"],
    }).json()
    assert next_page["items"][0]["revision"] == 2
    latest = client.get("/api/v1/claim-reviews").json()
    assert len(latest["items"]) == 1
    assert latest["items"][0]["status"] == "DISPUTED"
    assert latest["items"][0]["claim_id"] == claim
    assert client.post("/api/v1/claim-reviews", json={}).status_code == 405
    assert client.get("/api/v1/event-claims").status_code == 200
