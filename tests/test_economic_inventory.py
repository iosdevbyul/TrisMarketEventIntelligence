import sqlite3
from contextlib import closing

from tris_market_event_intelligence.economic_collection_cli import SCHEMA
from tris_market_event_intelligence.economic_inventory_cli import inventory


def test_inventory_counts_versions_without_claiming_vintage_or_coverage(tmp_path):
    macro = tmp_path / "macro.sqlite3"
    news = tmp_path / "news.sqlite3"
    with closing(sqlite3.connect(macro)) as db:
        db.executescript(SCHEMA)
        with db:
            db.executemany("INSERT INTO macro_observation VALUES (?,?,?,?,?,?,?,?)", [
                ("FRED_CSV", "CPI", "2025-01-01", "", "", "1", "now", "a"),
                ("FRED_CSV", "CPI", "2025-01-01", "", "", "2", "later", "b"),
            ])
    report = inventory(news, macro)
    assert report["numericSeriesCount"] == 1
    assert report["numericObservationRows"] == 2
    assert report["numeric"]["tables"]["series"][0]["periods"] == 1
    assert report["numeric"]["tables"]["series"][0]["unknown_vintage"] == 2
    assert report["releases"]["status"] == "MISSING_DATABASE"
    assert not news.exists()
    assert report["firstPublicAvailabilityVerified"] is False
