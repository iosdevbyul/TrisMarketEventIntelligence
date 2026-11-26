"""Retrospective target resolution using existing reviewed security identities."""
from __future__ import annotations

import argparse
import importlib
import json
import re
import unicodedata
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path

import httpx

from tris_market_event_intelligence.automatic_gemma_market_outcomes_cli import (
    evaluate_automatic_outcomes,
)
from tris_market_event_intelligence.historical_experience_learning import historical_rows
from tris_market_event_intelligence.listed_equity_price_coverage_cli import sha, write_csv
from tris_market_event_intelligence.market_instrument_master import acquire_master
from tris_market_event_intelligence.market_outcome_validation_cli import _csv, _mappings
from tris_market_event_intelligence.market_price_collection_cli import collect_prices
from tris_market_event_intelligence.market_price_recovery_cli import FIELDS, _observations
from tris_market_event_intelligence.market_proxy_cohort_cli import CSV_FIELDS
from tris_market_event_intelligence.research_strategy import coverage_diagnostics
from tris_market_event_intelligence.semantic_market_targets import block_equity, typed_mapping

# Identity hints only: a matching existing reviewed equity is mandatory.
ALIASES = {
    ("XNAS", "NVDA"): ("Nvidia", "NVIDIA Corp", "엔비디아"),
    ("XNAS", "META"): ("Meta Platforms", "Meta", "메타"),
    ("XNAS", "AAPL"): ("Apple Inc", "Apple", "애플"),
    ("XNAS", "MSFT"): ("Microsoft", "마이크로소프트"),
    ("XNAS", "AMD"): ("Advanced Micro Devices Inc (AMD)", "Advanced Micro Devices", "AMD"),
    ("XNAS", "AMZN"): ("Amazon", "Amazon.com", "아마존"),
    ("XKRX", "005930.KS"): ("삼성전자", "Samsung Electronics"),
    ("XKRX", "000660.KS"): ("SK하이닉스", "SK Hynix"),
}
INVALID = {"n/a", "none", "null", "unknown", "not applicable"}
PRIVATE = {"openai", "anthropic", "rockstar games"}


def normalize(name: str) -> str:
    value = unicodedata.normalize("NFKC", name).casefold().strip()
    value = re.sub(r"[,\s]+", " ", value)
    return re.sub(r"\s+(?:incorporated|inc|corporation|corp|limited|ltd)\.?$", "", value)


def expand_aliases(
    mappings: Path, report: dict, master: dict | None = None,
) -> tuple[list[dict], dict]:
    _mappings(mappings)
    rows = [
        {key: row.get(key, "") or "" for key in CSV_FIELDS}
        for row in _csv(mappings, set(CSV_FIELDS[:5]))
    ]
    quarantined = [row for row in rows if row["mapping_kind"] in {"", "DIRECT_EQUITY"}
                   and block_equity(row["target"])]
    rows = [row for row in rows if row not in quarantined]
    existing = {row["target"] for row in rows}
    index: dict[str, list[dict]] = {}
    for row in rows:
        if (row["mapping_kind"] not in {"", "DIRECT_EQUITY"}
                or not row["evidence_url"]):
            continue
        names = (row["target"], *ALIASES.get((row["market"], row["symbol"]), ()))
        for name in names:
            index.setdefault(normalize(name), []).append(row)
    master_index: dict[str, list[dict]] = {}
    if master is not None:
        for instrument in master["instruments"]:
            market, symbol = instrument["market"], instrument["symbol"]
            names = (instrument["name"], *ALIASES.get((market, symbol), ()))
            for name in names:
                master_index.setdefault(normalize(name), []).append(instrument)
    counts = Counter()
    unresolved = Counter()
    added = []
    now = datetime.now(UTC).isoformat()
    targets = Counter(
        item["target"] for item in report["items"] if item["status"] in {"UNMAPPED", "AMBIGUOUS_TARGET_SEMANTICS"}
    )
    for target, count in targets.items():
        if target in existing:
            continue
        typed = typed_mapping(target)
        if typed is not None:
            rows.append(typed)
            counts["RESOLVED_TYPED_MARKET_TARGET"] += count
            added.append({**typed, "impacts": count})
            continue
        if block_equity(target):
            counts["SEMANTIC_REVIEW_REQUIRED"] += count
            unresolved[target] += count
            continue
        name = normalize(target)
        if name in INVALID:
            counts["INVALID_TARGET"] += count
            continue
        if name in PRIVATE:
            counts["NO_DIRECT_LISTED_EQUITY_MAPPING"] += count
            continue
        matches = index.get(name, [])
        identities = {(r["market"], r["symbol"], r["benchmark"]) for r in matches}
        if not identities and master is not None:
            candidates = master_index.get(name, [])
            options = {(x["market"], x["symbol"]) for x in candidates}
            if len(options) == 1:
                instrument = candidates[0]
                rows.append({
                    "target": target, "market": instrument["market"],
                    "symbol": instrument["symbol"], "benchmark": instrument["benchmark"],
                    "verified_at": now, "mapping_kind": "DIRECT_EQUITY",
                    "proxy_limitation": "", "evidence_url": instrument["source"],
                })
                added.append({
                    "target": target, "anchorTarget": instrument["name"],
                    "symbol": instrument["symbol"], "impacts": count,
                    "evidenceUrl": instrument["source"],
                    "resolutionClass": "CURRENT_LISTING_RESEARCH_CANDIDATE",
                })
                counts["RESOLVED_CURRENT_LISTING_CANDIDATE"] += count
                continue
            if len(options) > 1:
                counts["AMBIGUOUS"] += count
                unresolved[target] += count
                continue
        if len(identities) != 1:
            counts["AMBIGUOUS" if identities else "UNRESOLVED"] += count
            unresolved[target] += count
            continue
        anchor = matches[0]
        rows.append({**anchor, "target": target, "verified_at": now})
        added.append({
            "target": target, "anchorTarget": anchor["target"],
            "symbol": anchor["symbol"], "impacts": count,
            "evidenceUrl": anchor["evidence_url"],
        })
        counts["RESOLVED_REVIEWED_ALIAS"] += count
    return rows, {
        "counts": dict(counts), "addedAliases": added,
        "quarantinedMappings": quarantined,
        "unresolvedTargets": dict(unresolved.most_common()),
        "newSecurityIdentitiesApproved": False,
        "retrospectiveOnly": True,
    }


def run(
    db: Path, mappings: Path, prices: Path, output: Path, *,
    model: str, fetch: bool, end: date, fetch_master: bool = False,
) -> dict:
    if output.exists() or output.is_symlink():
        raise ValueError("Output already exists")
    if end > datetime.now(UTC).date():
        raise ValueError("End must exclude unfinished UTC dates")
    before = evaluate_automatic_outcomes(
        db, mappings, prices, model=model, full_scan=True,
        price_source="Existing observed adjusted daily prices",
    )
    output.mkdir()
    master = None
    if fetch_master:
        with httpx.Client(timeout=30, trust_env=False) as client:
            master = acquire_master(output / "listing-snapshot", client)
    rows, resolution = expand_aliases(mappings, before, master)
    mapped = output / ("research-mappings.csv" if fetch_master else "reviewed-alias-mappings.csv")
    write_csv(mapped, CSV_FIELDS, rows)
    old = _observations(prices)
    merged = old.copy()
    collection = None
    if fetch:
        start = min(
            [date.fromisoformat(row["date"]) for group in old.values() for row in group]
            + [date.fromisoformat(item["publishedAt"][:10]) for item in before["items"]]
        )
        fresh, collection = collect_prices(
            mapped, start=start, end=end, client=importlib.import_module("yfinance"),
        )
        by_symbol: dict[str, list[dict]] = {}
        for row in fresh:
            by_symbol.setdefault(row["symbol"], []).append(row)
        # Never stitch old and fresh adjusted prices inside a symbol.
        for symbol, group in by_symbol.items():
            if symbol in old and (
                group[0]["date"] > old[symbol][0]["date"]
                or group[-1]["date"] < old[symbol][-1]["date"]
            ):
                continue
            merged[symbol] = group
    expanded = output / "expanded-adjusted-daily.csv"
    write_csv(expanded, FIELDS, [
        row for symbol in sorted(merged) for row in merged[symbol]
    ])
    after = evaluate_automatic_outcomes(
        db, mapped, expanded, model=model, full_scan=True,
        price_source="Observed Yahoo adjusted daily whole-symbol series",
    )
    result = {
        "status": "COMPLETED", "resolution": resolution, "collection": collection,
        "instrumentMaster": ({
            "observedAt": master["observedAt"], "sources": master["sources"],
            "failures": master["failures"], "instrumentCount": len(master["instruments"]),
            "historicalListingVerified": False,
        } if master is not None else None),
        "before": coverage_diagnostics(before, historical_rows(before)),
        "after": coverage_diagnostics(after, historical_rows(after)),
        "mappingSha256": sha(mapped), "pricesSha256": sha(expanded),
        "outputMappings": str(mapped), "outputPrices": str(expanded),
        "prospectivePredictionAccuracy": None, "promotionAllowed": False,
    }
    (output / "resolution-report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("db", "mappings", "prices", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument("--fetch-prices", action="store_true")
    parser.add_argument("--fetch-master", action="store_true")
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    result = run(
        args.db, args.mappings, args.prices, args.output_dir,
        model=args.model, fetch=args.fetch_prices, end=args.end, fetch_master=args.fetch_master,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
