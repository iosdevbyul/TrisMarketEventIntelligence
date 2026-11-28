import pytest

from tris_market_event_intelligence.listed_equity_price_coverage_cli import write_csv
from tris_market_event_intelligence.market_instrument_master import parse_krx
from tris_market_event_intelligence.market_proxy_cohort_cli import CSV_FIELDS
from tris_market_event_intelligence.market_target_resolution_cli import expand_aliases


def test_krx_listing_preserves_zeroes_and_market():
    raw = (
        "<table><tr><th>회사명</th><th>종목코드</th><th>상장일</th></tr>"
        "<tr><td>삼성전자</td><td>005930</td><td>1975-06-11</td></tr></table>"
    ).encode()
    result = parse_krx(raw, "stockMkt")
    assert result[0]["symbol"] == "005930.KS"
    assert result[0]["benchmark"] == "^KS11"
    assert parse_krx(raw, "kosdaqMkt")[0]["symbol"] == "005930.KQ"


def test_krx_rejects_nonlisting_response():
    with pytest.raises(ValueError, match="No valid KRX"):
        parse_krx(b"<html>Service unavailable</html>", "stockMkt")


def test_current_master_adds_new_research_identity(tmp_path):
    path = tmp_path / "mappings.csv"
    write_csv(path, CSV_FIELDS, [{
        "target": "Nvidia", "market": "XNAS", "symbol": "NVDA", "benchmark": "SPY",
        "verified_at": "2026-11-26T00:00:00+00:00", "mapping_kind": "DIRECT_EQUITY",
        "proxy_limitation": "", "evidence_url": "https://investor.nvidia.com/",
    }])
    master = {"instruments": [{
        "name": "Microsoft Corporation", "market": "XNAS", "symbol": "MSFT",
        "benchmark": "SPY", "source": "https://www.nasdaqtrader.com/",
    }]}
    rows, report = expand_aliases(path, {"items": [
        {"target": "Microsoft Corp", "status": "UNMAPPED"},
        {"target": "Anthropic", "status": "UNMAPPED"},
    ]}, master)
    assert rows[-1]["symbol"] == "MSFT"
    assert report["counts"]["RESOLVED_CURRENT_LISTING_CANDIDATE"] == 1
    assert report["newSecurityIdentitiesApproved"] is False
