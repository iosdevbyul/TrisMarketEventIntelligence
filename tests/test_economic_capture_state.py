import sqlite3

from tris_market_event_intelligence.global_economic_collection_cli import (
    CAPTURE_STATE_SCHEMA,
    capture_state,
)


def test_partial_history_survives_success():
    with sqlite3.connect(":memory:") as db:
        db.executescript(CAPTURE_STATE_SCHEMA)
        capture_state(db, 1, [{"error": "PDF"}])
        assert db.execute("SELECT status FROM official_release_capture_state").fetchone() == ("PARTIAL",)
        capture_state(db, 1, [])
        assert db.execute("SELECT status FROM official_release_capture_state").fetchone() == ("CAPTURED",)
        assert db.execute("SELECT status FROM official_release_capture_attempt ORDER BY id").fetchall() == [
            ("PARTIAL",), ("CAPTURED",),
        ]
