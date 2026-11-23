"""Conservative target *semantics* for real historical Gemma outcome gaps.

Never turn a rate/GDP/credit event or a private bank into an ETF merely
because a daily stock return is available. Explicit names below are
operator-observed UNMAPPED target strings, not market predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from tris_market_event_intelligence.market_outcome_gap_review_cli import QUEUE_COLUMNS
from tris_market_event_intelligence.market_outcome_validation_cli import _csv

MAX_BYTES = 8 * 1024 * 1024

# Verifiable entity relationship documents. These links identify a company,
# NOT an event's earliest release or a pre-news equity price forecast.
FED_NANO = "https://www.federalreserve.gov/newsevents/pressreleases/enforcement20261105a.htm"
FED_MUTUAL = "https://www.federalreserve.gov/newsevents/pressreleases/files/orders20261109a1.pdf"
FED_MAPLE = "https://www.federalreserve.gov/newsevents/pressreleases/files/orders20261029a1.pdf"
UCB_IR = "https://ir.ucbi.com/"

# Specific terms seen in the unlinked original Gemma batch. New/unrecognized
# names fail closed to UNCLASSIFIED, rather than receiving a guessed ETF.
# Fields are category, observability, candidate parent (not approval), evidence.
KNOWN: dict[str, tuple[str, str, str, str]] = {
    "Financial Markets": ("BROAD_MARKET", "REQUIRES_REGION_AND_INDEX", "", ""),
    "Allegiant United Holdings, LLC, Nano Financial Holdings, Inc., Nano Banc": (
        "MULTI_PRIVATE_OR_UNCONFIRMED_ENTITY", "NO_SINGLE_LISTED_SERIES", "", FED_NANO,
    ),
    "Commercial Lending": ("ECONOMIC_ACTIVITY", "CREDIT_VOLUME_OR_LENDING_SERIES", "", ""),
    "Consumer Sectors": ("BROAD_SECTOR", "REQUIRES_REGION_AND_COMPONENTS", "", ""),
    "Corporate Bonds/Credit Markets": ("CREDIT_MARKET", "CREDIT_SPREAD_OR_BOND_INDEX", "", ""),
    "Economic Growth (GDP)": ("MACRO_RELEASE", "GDP_RELEASE_OR_SURPRISE", "", ""),
    "Economic Growth / Corporate Investment": (
        "MULTI_MACRO", "SEPARATE_GDP_AND_INVESTMENT_SERIES", "", "",
    ),
    "Equity Markets": ("BROAD_MARKET", "REQUIRES_REGION_AND_INDEX", "", ""),
    "Equity Markets (South Korea, Taiwan)": (
        "MULTI_MARKET", "SEPARATE_KRX_AND_TWSE_BENCHMARKS", "", "",
    ),
    "Financial Institutions (Banks/Credit Unions)": (
        "MIXED_INSTITUTIONS", "BANK_AND_CREDIT_UNION_SERIES_NOT_ONE_STOCK", "", "",
    ),
    "Financial Markets, Interest Rates": (
        "MULTI_VARIABLE", "SEPARATE_MARKET_INDEX_AND_YIELD", "", "",
    ),
    "Financials, Interest-Rate Sensitive Sectors": (
        "MULTI_SECTOR", "SPECIFY_COMPONENTS_AND_GEOGRAPHY", "", "",
    ),
    "Illinois Banking Sector": (
        "REGIONAL_SECTOR", "REGIONAL_BANK_INDEX_NEEDS_EXPOSURE_EVIDENCE", "", "",
    ),
    "Inflation": ("MACRO_RELEASE", "CPI_OR_PCE_RELEASE_WITH_JURISDICTION", "", ""),
    "Inflation, Energy Sectors": (
        "MULTI_VARIABLE", "SEPARATE_INFLATION_AND_ENERGY", "", "",
    ),
    "Inflation-Sensitive Sectors (Energy/Commodities)": (
        "MULTI_SECTOR", "SEPARATE_ENERGY_AND_COMMODITY_BENCHMARKS", "", "",
    ),
    "Interest Rates": ("YIELD_SERIES", "DEFINE_CURRENCY_TENOR_AND_INSTRUMENT", "", ""),
    "Interest Rates / Treasury Markets": (
        "YIELD_SERIES", "DEFINE_TREASURY_TENOR_AND_PRICE_VS_YIELD", "", "",
    ),
    "MF Bank": ("BANK_SUBSIDIARY", "NO_DIRECT_EQUITY_ASSUMED", "", FED_MUTUAL),
    "MFB Bancorp, Inc. (MFB Bancorp)": (
        "CONVERSION_ENTITY", "NO_LISTING_ASSUMED_ON_PUBLICATION_DATE", "", FED_MUTUAL,
    ),
    "MapleMark": (
        "ACQUISITION_TARGET_BANK", "ACQUIRER_EQUITY_IS_DIFFERENT_EXPOSURE", "", FED_MAPLE,
    ),
    "Mutual Federal Bancorp, MHC (MFB MHC)": (
        "MUTUAL_HOLDING_COMPANY", "NO_DIRECT_EQUITY_ASSUMED", "", FED_MUTUAL,
    ),
    "Stock Market": ("BROAD_MARKET", "REQUIRES_REGION_AND_INDEX", "", ""),
    "Treasury Yields": ("YIELD_SERIES", "DEFINE_CURRENCY_TENOR_AND_INSTRUMENT", "", ""),
    "United Community Bank": (
        "LISTED_PARENT_CANDIDATE", "PARENT_EQUITY_NEEDS_EVENT_REVIEW", "UCB", UCB_IR,
    ),
}
DIRECT_EQUITY_REVIEW_KINDS = frozenset({"LISTED_PARENT_CANDIDATE"})
INDEX_PROXY_REVIEW_KINDS = frozenset({"BROAD_SECTOR", "REGIONAL_SECTOR"})
NO_STOCK_PROXY = frozenset({
    "BROAD_MARKET", "MULTI_MARKET", "MACRO_RELEASE", "MULTI_MACRO",
    "YIELD_SERIES", "CREDIT_MARKET", "ECONOMIC_ACTIVITY", "MULTI_VARIABLE",
    "MIXED_INSTITUTIONS", "MULTI_SECTOR",
    "MULTI_PRIVATE_OR_UNCONFIRMED_ENTITY", "BANK_SUBSIDIARY",
    "CONVERSION_ENTITY", "MUTUAL_HOLDING_COMPANY", "ACQUISITION_TARGET_BANK",
    "UNCLASSIFIED",
})


def _sha(path: Path) -> str:
    if (not path.is_file() or path.is_symlink()
            or path.stat().st_size > MAX_BYTES):
        raise ValueError("Bounded regular input required")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze_target_semantics(
    gap_report: Path, original_queue: Path,
) -> dict[str, object]:
    source_sha = _sha(gap_report)
    review_sha = _sha(original_queue)
    gap = json.loads(gap_report.read_text(encoding="utf-8"))
    if (gap.get("schemaVersion") != "market_outcome_gap_review_v1"
            or gap.get("mappingApprovedAutomatically") is not False
            or gap.get("trainingEligible") is not False):
        raise ValueError("Expected immutable research-only gap report")
    original = gap.get("mappingQueue")
    rows = _csv(original_queue, set(QUEUE_COLUMNS))
    if not isinstance(original, list) or len(rows) != len(original):
        raise ValueError("Mismatch between original queue and gap report")
    seen: set[str] = set()
    classified = []
    counts: Counter[str] = Counter()
    for item in rows:
        target = item["target"]
        if not isinstance(target, str) or not target.strip() or target in seen:
            raise ValueError("Malformed or duplicate target")
        seen.add(target)
        matching = next((x for x in original if x.get("target") == target), None)
        if matching is None:
            raise ValueError("Unexpected target in review")
        for key in ("target", "impacts", "articles", "article_ids", "predictions", "issue"):
            if str(matching[key]) != item[key]:
                raise ValueError("Review queue original evidence changed")
        kind, observable, equity, evidence = KNOWN.get(
            target, ("UNCLASSIFIED", "PRIMARY_RESEARCH_REQUIRED", "", ""),
        )
        counts[kind] += 1
        classified.append({
            "target": target, "articleIds": item["article_ids"],
            "impactCount": int(item["impacts"]),
            "targetKind": kind, "measurementRequirement": observable,
            "candidateRelatedListedParentSymbol": equity or None,
            "candidateEntityRelationshipEvidence": evidence or None,
            "securityMappingApproved": False,
            "reviewDecision": "NEEDS_PRIMARY_EVIDENCE",
            "legacyStockReturnEligible": False,
        })
    classified.sort(key=lambda x: (-x["impactCount"], x["target"]))
    if sum(x["impactCount"] for x in classified) != gap["unmappedImpacts"]:
        raise ValueError("Gap impact count and reviewed queue disagree")
    return {
        "schemaVersion": "typed_market_target_inventory_v1",
        "gapSha256": source_sha, "queueSha256": review_sha,
        "targetCount": len(classified),
        "impactCount": sum(x["impactCount"] for x in classified),
        "classificationCounts": dict(sorted(counts.items())),
        "targets": classified,
        "anyStockMappingApproved": False,
        "anyEconomicSeriesVerified": False,
        "newMarketPriceObservations": 0,
        "trainingEligible": False,
        "prospectiveAccuracy": None,
        "measurementPolicy": (
            "A category is not a measured series. Multi-target claims require "
            "component-level separately timestamped observations. Macro GDP/CPI "
            "and yields require their own units and publication/revision contracts; "
            "they must never be silently scored as an arbitrary ETF."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-only real target type inventory")
    parser.add_argument("--gap-report", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("Output exists; immutable results must be preserved")
        result = analyze_target_semantics(args.gap_report, args.review_csv)
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(json.dumps({
            "targets": result["targetCount"],
            "impacts": result["impactCount"],
            "classificationCounts": result["classificationCounts"],
            "mappingApproved": False,
            "output": str(args.output),
        }, ensure_ascii=False))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"typed-market-targets: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
