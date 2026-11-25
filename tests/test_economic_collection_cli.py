import sqlite3
from datetime import date

import httpx

from tris_market_event_intelligence.economic_collection_cli import (
    SCHEMA,
    bls_collect,
    fred_collect,
    fred_csv_collect,
    korea_world_bank_collect,
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


def test_fred_csv_without_key_and_without_vintage():
    def handler(request):
        assert "api_key" not in request.url.params
        return httpx.Response(
            200, text="observation_date,X\n2020-01-01,10\n2020-02-01,.\n2019-12-01,8\n"
        )

    with sqlite3.connect(":memory:") as db:
        db.executescript(SCHEMA)
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            args = (client, db, "X", date(2020, 1, 1), date(2020, 12, 31))
            assert fred_csv_collect(*args) == 1
            assert fred_csv_collect(*args) == 0
        assert db.execute(
            "SELECT source, vintage_start, vintage_end, value FROM macro_observation"
        ).fetchone() == ("FRED_CSV", "", "", "10")


def test_korea_world_bank_keyless_collection():
    def handler(request):
        assert "/country/KOR/indicator/X" in request.url.path
        assert "api_key" not in request.url.params
        page = int(request.url.params["page"])
        return httpx.Response(200, json=[
            {"page": page, "pages": 2, "total": 2},
            [{
                "countryiso3code": "KOR", "indicator": {"id": "X"},
                "date": str(2020 + page), "value": 2.5 if page == 1 else None,
            }],
        ])

    with sqlite3.connect(":memory:") as db:
        db.executescript(SCHEMA)
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            args = (client, db, "X", date(2020, 1, 1), date(2022, 12, 31))
            assert korea_world_bank_collect(*args) == 1
            assert korea_world_bank_collect(*args) == 0
        assert db.execute(
            "SELECT source, period, vintage_start, value FROM macro_observation"
        ).fetchone() == ("WORLD_BANK_KOR", "2021-01-01", "", "2.5")
