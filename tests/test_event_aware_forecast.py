import hashlib
import json
import sqlite3

import httpx
import pytest

from tris_market_event_intelligence.automatic_forecast_precommitment import (
    initialize,
    record_new_analysis,
    verify_chain,
)
from tris_market_event_intelligence.event_aware_forecast import (
    VERSION,
    _response_content,
    horizon_prediction,
    prior_context,
    review_dual,
    validate_dual,
)


def answer():
    return {
        "summary": "Contract",
        "limitations": "Uncertain expectations",
        "novelty": "UNKNOWN",
        "related_article_ids": [],
        "novelty_reason": "No earlier identified event", "new_fact": "NONE",
        "impacts": [{
            "target": "Nvidia", "direction": "DOWN",
            "reason": "Near-term costs", "evidence_id": "E1",
            "economic_direction": "POSITIVE", "economic_reason": "Future revenue",
            "direction_5d": "UP", "reason_5d": "Longer-term revenue",
            "forward_mechanism": "Near-term costs",
            "forward_mechanism_5d": "Longer-term revenue", "evidence_id_5d": "E1",
        }],
    }


def test_dual_forecast_preserves_opposing_horizons_in_existing_chain():
    data = validate_dual(json.dumps(answer()), {"E1": "Contract signed."}, [])
    assert data["forecastVersion"] == VERSION
    assert horizon_prediction(data, data["impacts"][0], 1)[0] == "DOWN"
    assert horizon_prediction(data, data["impacts"][0], 5)[0] == "UP"
    with sqlite3.connect(":memory:") as db:
        initialize(db)
        record_new_analysis(
            db, article={
                "articleId": 1, "publishedAt": "2026-11-23T13:00:00+00:00",
                "detectedAt": "2026-11-23T13:01:00+00:00",
                "bodyObservedAt": "2026-11-23T13:02:00+00:00",
            },
            analysis=data, model="gemma4:e2b", body_sha256="a" * 64,
            analyzed_at="2026-11-23T13:03:00+00:00", mappings={},
        )
        assert db.execute(
            "SELECT horizon_sessions,direction FROM automatic_forecast_commitments "
            "ORDER BY horizon_sessions"
        ).fetchall() == [(1, "DOWN"), (5, "UP")]
        assert verify_chain(db)["records"] == 2


def test_follow_up_requires_admitted_context():
    data = answer()
    data["novelty"] = "FOLLOW_UP_CANDIDATE"
    data["related_article_ids"] = [12]
    with pytest.raises(ValueError, match="RELATED_ARTICLE_NOT_IN_CONTEXT"):
        validate_dual(json.dumps(data), {"E1": "Contract signed."}, [])


def test_legacy_horizon_unchanged():
    impact = {"direction": "UNCERTAIN", "reason": "Missing evidence"}
    assert horizon_prediction({}, impact, 5) == ("UNCERTAIN", "Missing evidence")


def test_duplicate_prior_bodies_keep_earliest_observation():
    body = "Company contract signed."
    digest = hashlib.sha256(body.encode()).hexdigest()
    with sqlite3.connect(":memory:") as db:
        db.executescript("""
            CREATE TABLE articles(id INTEGER, title TEXT, published_at TEXT, detected_at TEXT);
            CREATE TABLE article_full_text(article_id INTEGER, observed_at TEXT,
                body TEXT, body_sha256 TEXT, status TEXT);
        """)
        for i, minute in [(1, "01"), (2, "02")]:
            clock = "2026-11-23T12:" + minute + ":00+00:00"
            db.execute("INSERT INTO articles VALUES (?,?,?,?)",
                       (i, "Company contract signed", clock, clock))
            db.execute("INSERT INTO article_full_text VALUES (?,?,?,?,?)",
                       (i, clock, body, digest, "COMPLETE"))
        context = prior_context(db, {
            "articleId": 3, "title": "Company contract signed again",
            "publishedAt": "2026-11-23T13:00:00+00:00",
        })
        assert [item["articleId"] for item in context] == [1]


def test_new_fact_cannot_ignore_related_candidates():
    data = answer()
    data["novelty"] = "NEW_FACT_CANDIDATE"
    with pytest.raises(ValueError, match="NEW_FACT_WITHOUT_CONTEXT_COMPARISON"):
        validate_dual(json.dumps(data), {"E1": "Contract signed."}, [{"articleId": 12}])


def test_review_rejects_unsupported_future_mechanism():
    def handler(request):
        return httpx.Response(200, json={"message": {"content": json.dumps({
            "event_relation_supported": True, "source_claims_supported": True,
            "forward_mechanisms_supported": False,
        })}})
    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(ValueError, match="SOURCE_GROUNDING_REVIEW_REJECTED"),
    ):
        review_dual(client, "gemma4:e2b", answer(), [], [])


def test_truncated_review_has_stage_specific_error():
    response = httpx.Response(200, json={
        "done_reason": "length", "message": {"content": '{"event_relation_supported":'},
    })
    with pytest.raises(ValueError, match="REVIEW_GENERATION_TRUNCATED"):
        _response_content(response, "REVIEW")


def test_invalid_review_json_is_not_accepted():
    def handler(request):
        return httpx.Response(200, json={"message": {"content": "not JSON"}})

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(ValueError, match="REVIEW_JSON_INVALID"),
    ):
        review_dual(client, "gemma4:e2b", answer(), [], [])
