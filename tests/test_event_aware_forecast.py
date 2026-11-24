import json
import sqlite3

import pytest

from tris_market_event_intelligence.automatic_forecast_precommitment import (
    initialize,
    record_new_analysis,
    verify_chain,
)
from tris_market_event_intelligence.event_aware_forecast import (
    VERSION,
    horizon_prediction,
    validate_dual,
)


def answer():
    return {
        "summary": "Contract",
        "limitations": "Uncertain expectations",
        "novelty": "UNKNOWN",
        "related_article_ids": [],
        "impacts": [{
            "target": "Nvidia", "direction": "DOWN",
            "reason": "Near-term costs", "evidence_id": "E1",
            "economic_direction": "POSITIVE", "economic_reason": "Future revenue",
            "direction_5d": "UP", "reason_5d": "Longer-term revenue",
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
