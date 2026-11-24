import json
from datetime import date

import httpx

from tris_market_event_intelligence.collector import NewsStore
from tris_market_event_intelligence.sec_operational_discovery_cli import discover


def test_discovery_all_forms_and_idempotence(tmp_path, monkeypatch):
    monkeypatch.setattr("tris_market_event_intelligence.sec_operational_discovery_cli.time.sleep", lambda _: None)
    packet = {"cik": "789019", "name": "Microsoft", "filings": {"files": [], "recent": {
        "accessionNumber": ["0000789019-26-000001", "0000789019-26-000002", "0000789019-26-000003"],
        "filingDate": ["2026-11-20"] * 3,
        "acceptanceDateTime": ["2026-11-20T12:00:00Z"] * 3,
        "form": ["8-K", "10-Q", "4"],
    }}}
    store = NewsStore(tmp_path / "news.sqlite3")
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=json.dumps(packet).encode())
    )) as client:
        first = discover(store, client, cik=789019, start=date(2026, 11, 20), end=date(2026, 11, 20))
        second = discover(store, client, cik=789019, start=date(2026, 11, 20), end=date(2026, 11, 20))
    assert first["inserted"] == 3
    assert first["forms"] == {"8-K": 1, "10-Q": 1, "4": 1}
    assert second["inserted"] == 0
    assert second["alreadyKnown"] == 3
