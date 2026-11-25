import sqlite3
from datetime import date

import httpx

from tris_market_event_intelligence.economic_collection_cli import (
    SCHEMA,
    bls_collect,
    fred_collect,
    save,
)


def test_save_retains_revisions_and_deduplicates():
    with sqlite3.connect(":memory:") as db:
        db.executescript(SCHEMA)
        first = [("2020-01-01", "2020-02-01", "2020-03-01", "10")]
        second = [("2020-01-01", "2020-03-02", "2020-04-01", "11")]
        assert save(db, "FRED", "X", {"rows": first}, first) == 1
        assert save(db, "FRED", "X", {"rows": first}, first) == 0
        assert save(db, "FRED", "X", {"rows": second}, second) == 1
        assert db.execute("SELECT COUNT(*) FROM macro_raw").fetchone()[0] == 2


def test_fred_pagination():
    def handler(request):
        offset = int(request.url.params["offset"])
        return httpx.Response(200, json={
            "count": 2,
            "observations": [{
                "date": f"2020-0{offset + 1}-01",
                "realtime_start": "2020-03-01",
                "realtime_end": "2020-04-01",
                "value": "10",
            }],
        })

    with sqlite3.connect(":memory:") as db:
        db.executescript(SCHEMA)
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            assert fred_collect(
                client, db, "X", date(2020, 1, 1), date(2020, 12, 31), "test"
            ) == 2


def test_bls_does_not_invent_vintage():
    def handler(request):
        return httpx.Response(200, json={
            "status": "REQUEST_SUCCEEDED",
            "Results": {"series": [{"seriesID": "X", "data": [
                {"year": "2020", "period": "M01", "value": "10"},
                {"year": "2020", "period": "M13", "value": "11"},
            ]}]},
        })

    with sqlite3.connect(":memory:") as db:
        db.executescript(SCHEMA)
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            assert bls_collect(client, db, "X", date(2020, 1, 1), date(2020, 12, 31)) == 1
        assert db.execute(
            "SELECT vintage_start, vintage_end FROM macro_observation"
        ).fetchone() == ("", "")
