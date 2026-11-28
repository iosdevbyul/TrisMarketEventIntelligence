from tris_market_event_intelligence.listed_equity_price_coverage_cli import write_csv
from tris_market_event_intelligence.market_outcome_validation_cli import _mappings
from tris_market_event_intelligence.market_proxy_cohort_cli import CSV_FIELDS
from tris_market_event_intelligence.market_target_resolution_cli import expand_aliases
from tris_market_event_intelligence.semantic_market_targets import block_equity, typed_mapping


def test_quarantine_existing_generic_targets(tmp_path):
    path = tmp_path / "old.csv"
    row = {
        "target": "Financial Institutions", "market": "XNAS", "symbol": "FISI",
        "benchmark": "SPY", "verified_at": "2026-11-26T00:00:00+00:00",
        "mapping_kind": "DIRECT_EQUITY", "proxy_limitation": "",
        "evidence_url": "https://www.nasdaqtrader.com/",
    }
    write_csv(path, CSV_FIELDS, [row])
    rows, report = expand_aliases(path, {"items": [
        {"target": "Financial Institutions", "status": "AMBIGUOUS_TARGET_SEMANTICS"},
        {"target": "코스피", "status": "UNMAPPED"},
    ]})
    assert report["quarantinedMappings"] == [row]
    assert len(rows) == 1
    assert rows[0]["symbol"] == "^KS11"
    output = tmp_path / "new.csv"
    write_csv(output, CSV_FIELDS, rows)
    assert _mappings(output)["코스피"]["mappingKind"] == "MARKET_INDEX"


def test_ambiguous_market_and_unqualified_sector():
    assert block_equity("Nasdaq")
    assert block_equity("DANA")
    assert block_equity("Semiconductor Sector")
    assert not block_equity("Nasdaq, Inc.")
    assert typed_mapping("Semiconductor Sector") is None
    assert typed_mapping("Nasdaq") is None
    assert typed_mapping("US defense sector")["mapping_kind"] == "SECTOR_ETF_PROXY"
