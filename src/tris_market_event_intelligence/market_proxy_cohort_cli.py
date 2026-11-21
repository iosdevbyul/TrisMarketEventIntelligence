"""Create a transparent research-only ETF proxy cohort from original Gemma targets.

Explicit allowlist: never infer tradable symbols from arbitrary LLM text.
Generated links are retrospectively verified, not publication-time predictions.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.market_outcome_validation_cli import _load_json

SPDR_EVIDENCE = (
    "https://www.ssga.com/us/en/individual/capabilities/"
    "equities/sector-investing/select-sector-etfs"
)
PROXY_CATALOG = {
    "Technology Sector": (
        "XLK", "S&P 500 US technology sector; excludes some technology companies.",
        SPDR_EVIDENCE,
    ),
    "Technology/AI Sector": (
        "XLK", "US technology proxy, not a comprehensive global AI equity portfolio.",
        SPDR_EVIDENCE,
    ),
    "Oil and Energy Sector": (
        "XLE", "US S&P 500 energy equities, not oil commodity prices or global energy.",
        SPDR_EVIDENCE,
    ),
    "Energy Sector": (
        "XLE", "US S&P 500 energy equities, not global energy equities or commodities.",
        SPDR_EVIDENCE,
    ),
    "Banking Sector": (
        "XLF", "Broad US financial sector including nonbanks; not a pure bank index.",
        SPDR_EVIDENCE,
    ),
    "Banking Sector (U.S.)": (
        "XLF", "Broad US financial sector including nonbanks; not a pure bank index.",
        SPDR_EVIDENCE,
    ),
    "Financial Services Sector": (
        "IYG", "US listed financial services, not worldwide financial services.",
        "https://www.ishares.com/us/products/239509/ishares-us-financial-services-etf",
    ),
    "Payment Services Sector": (
        "IPAY", "Digital payments equities only; excludes many physical payment services.",
        "https://amplifyetfs.com/ipay/",
    ),
}
DIRECT_EQUITIES = {
    "The Bank of Nova Scotia (BNS)": (
        "BNS",
        "https://www.sec.gov/Archives/edgar/data/9631/000119312526364091/d143422dex991.htm",
    ),
}

CSV_FIELDS = [
    "target", "market", "symbol", "benchmark", "verified_at",
    "mapping_kind", "proxy_limitation", "evidence_url",
]


def prepare_cohort(
    articles_dir: Path, *, verified_at: datetime,
) -> tuple[list[dict[str, str]], dict]:
    if verified_at.utcoffset() is None:
        raise ValueError("Mapping verification time must include a timezone")
    files = sorted(articles_dir.glob("*.json"))
    if not files or not articles_dir.is_dir() or articles_dir.is_symlink():
        raise ValueError("Gemma JSON results directory is required")
    counts: Counter[str] = Counter()
    articles = set()
    for path in files:
        report = _load_json(path)
        if report.get("schemaVersion") != "gemma_original_article_review_v2":
            raise ValueError("Unexpected article source schema")
        articles.add(report["article"]["articleId"])
        for impact in report["analysis"]["impacts"]:
            counts[impact["target"].strip()] += 1
    rows = []
    omitted = []
    for target, count in sorted(counts.items()):
        if target in DIRECT_EQUITIES:
            symbol, evidence = DIRECT_EQUITIES[target]
            kind, limitation = "DIRECT_EQUITY", ""
        elif target in PROXY_CATALOG:
            symbol, limitation, evidence = PROXY_CATALOG[target]
            kind = "SECTOR_ETF_PROXY"
        else:
            omitted.append({
                "target": target, "impactCount": count,
                "reason": "NO_DEFENSIBLE_SINGLE_EQUITY_PRICE_MAPPING",
            })
            continue
        rows.append({
            "target": target, "market": "XNYS", "symbol": symbol,
            "benchmark": "SPY", "verified_at": verified_at.isoformat(),
            "mapping_kind": kind, "proxy_limitation": limitation,
            "evidence_url": evidence,
        })
    report = {
        "schemaVersion": "market_proxy_mapping_review_v1",
        "researchOnly": True,
        "mappingVerifiedAt": verified_at.isoformat(),
        "predictionSealedBeforeOutcome": False,
        "articlesInspected": len(articles),
        "impactsInspected": sum(counts.values()),
        "distinctTargets": len(counts),
        "mappedTargets": len(rows),
        "mappedImpacts": sum(counts[row["target"]] for row in rows),
        "omitted": omitted,
        "method": "Exact-string predeclared source-backed proxy allowlist; no LLM ticker guessing",
        "limitations": [
            "Research-only unsealed retrospective outcome comparison",
            "ETF proxies are not the precise market or sector named by Gemma",
            "Historical adjusted close does not attribute causation to an article",
        ],
    }
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemma-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.output.with_name(args.output.name + ".mapping-review.json")
    if any(path.exists() or path.is_symlink() for path in (args.output, report_path)):
        parser.error("Mapping or review report already exists")
    try:
        rows, review = prepare_cohort(args.gemma_dir, verified_at=datetime.now(UTC))
        if not rows:
            raise ValueError("No allowlisted source-backed mapping matched current articles")
        with args.output.open("x", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        with report_path.open("x", encoding="utf-8") as output:
            json.dump(review, output, ensure_ascii=False, indent=2)
            output.write("\n")
        print(json.dumps({
            "mappedTargets": review["mappedTargets"],
            "mappedImpacts": review["mappedImpacts"],
            "impactsInspected": review["impactsInspected"],
            "omittedTargets": len(review["omitted"]),
            "output": str(args.output), "review": str(report_path),
        }, ensure_ascii=False))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"market-proxy-cohort: {exc}\n")


if __name__ == "__main__":
    main()
