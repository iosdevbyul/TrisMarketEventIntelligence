from tris_market_event_intelligence.listed_equity_price_coverage_cli import write_csv
from tris_market_event_intelligence.market_proxy_cohort_cli import CSV_FIELDS
from tris_market_event_intelligence.market_target_resolution_cli import expand_aliases


def test_resolve_only_reviewed_identity(tmp_path):
    mappings = tmp_path / "mappings.csv"
    anchor = {
        "target": "Nvidia", "market": "XNAS", "symbol": "NVDA", "benchmark": "SPY",
        "verified_at": "2026-11-26T00:00:00+00:00", "mapping_kind": "DIRECT_EQUITY",
        "proxy_limitation": "", "evidence_url": "https://investor.nvidia.com/",
    }
    write_csv(mappings, CSV_FIELDS, [anchor])
    report = {"items": [
        {"target": target, "status": "UNMAPPED"}
        for target in ("NVIDIA CORP", "엔비디아", "Anthropic", "N/A", "Microsoft")
    ]}
    rows, diagnostics = expand_aliases(mappings, report)
    assert rows[0] == anchor
    assert {row["target"] for row in rows[1:]} == {"NVIDIA CORP", "엔비디아"}
    assert all(row["symbol"] == "NVDA" for row in rows)
    assert diagnostics["counts"] == {
        "RESOLVED_REVIEWED_ALIAS": 2, "NO_DIRECT_LISTED_EQUITY_MAPPING": 1,
        "INVALID_TARGET": 1, "UNRESOLVED": 1,
    }


def test_ambiguous_normalized_name_is_not_mapped(tmp_path):
    mappings = tmp_path / "mappings.csv"
    anchor = {
        "target": "Example Inc", "market": "XNAS", "symbol": "EXA", "benchmark": "SPY",
        "verified_at": "2026-11-26T00:00:00+00:00", "mapping_kind": "DIRECT_EQUITY",
        "proxy_limitation": "", "evidence_url": "https://example.com/",
    }
    write_csv(mappings, CSV_FIELDS, [
        anchor, {**anchor, "target": "Example Corp", "symbol": "EXB"},
    ])
    rows, diagnostics = expand_aliases(
        mappings, {"items": [{"target": "Example", "status": "UNMAPPED"}]},
    )
    assert len(rows) == 2
    assert diagnostics["counts"] == {"AMBIGUOUS": 1}
