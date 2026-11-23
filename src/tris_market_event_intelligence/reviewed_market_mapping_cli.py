"""Apply explicitly reviewed market target mappings without guessing securities.

Consumes the immutable gap report and the operator-edited review queue.  A
signed-looking SHA or reviewer date does not independently attest an exchange
listing or historical availability.  Research remains retrospective.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

from tris_market_event_intelligence.market_outcome_validation_cli import (
    ZONES,
    _csv,
    _mappings,
    _timestamp,
)
from tris_market_event_intelligence.market_price_recovery_cli import _observations
from tris_market_event_intelligence.market_proxy_cohort_cli import CSV_FIELDS
from tris_market_event_intelligence.typed_market_targets_cli import KNOWN

REVIEW_ORIGINAL_FIELDS = (
    "target", "impacts", "articles", "article_ids", "predictions", "issue",
)
REVIEW_EDIT_FIELDS = (
    "review_decision", "reviewed_market", "reviewed_symbol",
    "reviewed_benchmark", "mapping_kind", "mapping_evidence_url",
    "mapping_verified_at", "proxy_limitation",
)
DECISIONS = {"PENDING", "NOT_REPRESENTABLE", "APPROVE"}
SYMBOL = re.compile(r"^[A-Za-z0-9.^_-]{1,32}$")
MAX_BYTES = 8 * 1024 * 1024


def _hash(path: Path) -> str:
    if (not path.is_file() or path.is_symlink()
            or path.stat().st_size > MAX_BYTES):
        raise ValueError("Expected existing bounded non-symlink input")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _review_rows(path: Path) -> list[dict[str, str]]:
    _hash(path)
    rows = _csv(path, set(REVIEW_ORIGINAL_FIELDS + REVIEW_EDIT_FIELDS))
    if len(rows) > 2000 or any(
        None in row or any(value is None for value in row.values())
        for row in rows
    ):
        raise ValueError("Malformed or excessive reviewed target CSV rows")
    return rows


def _evidence_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return (parsed.scheme == "https" and parsed.username is None
            and parsed.password is None and bool(parsed.hostname)
            and "." in parsed.hostname and not any(x.isspace() for x in value))


def _validate_approval(row: dict[str, str]) -> dict[str, str]:
    market = row["reviewed_market"].strip()
    symbol = row["reviewed_symbol"].strip()
    benchmark = row["reviewed_benchmark"].strip()
    kind = row["mapping_kind"].strip()
    evidence = row["mapping_evidence_url"].strip()
    verified = row["mapping_verified_at"].strip()
    limitation = row["proxy_limitation"].strip()
    if (market not in ZONES or SYMBOL.fullmatch(symbol) is None
            or SYMBOL.fullmatch(benchmark) is None or symbol == benchmark
            or kind not in {"DIRECT_EQUITY", "SECTOR_ETF_PROXY"}
            or not _evidence_url(evidence) or not verified):
        raise ValueError(f"Unsubstantiated or invalid approved target: {row['target']}")
    _timestamp(verified)
    if kind == "SECTOR_ETF_PROXY" and not limitation:
        raise ValueError(f"Sector proxy requires explicit limitation: {row['target']}")
    return {
        "target": row["target"], "market": market,
        "symbol": symbol, "benchmark": benchmark,
        "verified_at": verified, "mapping_kind": kind,
        "proxy_limitation": limitation, "evidence_url": evidence,
    }


def apply_review(
    gap_report: Path,
    source_outcomes: Path,
    reviewed_csv: Path,
    original_mappings: Path,
    price_csv: Path,
    *, semantic_plan: Path | None = None,
) -> tuple[dict[str, object], list[dict[str, str]], list[dict[str, str]]]:
    """Build explicit mapping/price-fetch plan without modifying any source."""
    for path in (gap_report, source_outcomes, reviewed_csv, original_mappings, price_csv):
        _hash(path)
    gap = json.loads(gap_report.read_text(encoding="utf-8"))
    if (gap.get("schemaVersion") != "market_outcome_gap_review_v1"
            or gap.get("inputSha256") != _hash(source_outcomes)
            or gap.get("trainingEligible") is not False
            or gap.get("mappingApprovedAutomatically") is not False):
        raise ValueError("Gap report and immutable source outcome do not match")
    queue = gap.get("mappingQueue")
    if not isinstance(queue, list) or len(queue) > 2000:
        raise ValueError("Invalid original unapproved mapping queue")
    rows = _review_rows(reviewed_csv)
    semantic_by_target: dict[str, dict] = {}
    if semantic_plan is not None:
        _hash(semantic_plan)
        semantics = json.loads(semantic_plan.read_text(encoding="utf-8"))
        if (semantics.get("schemaVersion") != "typed_market_target_inventory_v1"
                or semantics.get("gapSha256") != _hash(gap_report)
                or semantics.get("anyStockMappingApproved") is not False
                or semantics.get("trainingEligible") is not False):
            raise ValueError("Semantic plan provenance or policy mismatch")
        records = semantics.get("targets")
        if not isinstance(records, list):
            raise ValueError("Malformed semantic target list")
        for record in records:
            if (not isinstance(record, dict)
                    or not isinstance(record.get("target"), str)
                    or record["target"] in semantic_by_target):
                raise ValueError("Duplicate or malformed semantic target")
            semantic_by_target[record["target"]] = record
        if set(semantic_by_target) != {
            item["target"] for item in gap["mappingQueue"]
        }:
            raise ValueError("Semantic plan target scope does not match gap")
    if len(rows) != len(queue):
        raise ValueError("Reviewed queue must contain every original target")
    expected = {}
    for item in queue:
        if not isinstance(item, dict) or item.get("target") in expected:
            raise ValueError("Duplicate source queue target")
        expected[item["target"]] = item
    reviewed = set()
    approved = []
    deferred = []
    rejected = []
    for row in rows:
        target = row["target"]
        if target not in expected or target in reviewed:
            raise ValueError("Unexpected or duplicate target in review")
        reviewed.add(target)
        source = expected[target]
        for name in REVIEW_ORIGINAL_FIELDS:
            if str(source[name]) != row[name]:
                raise ValueError(f"Original review evidence was edited: {target} {name}")
        decision = row["review_decision"].strip().upper()
        if decision not in DECISIONS:
            raise ValueError(f"Unknown review decision for {target}")
        if decision == "APPROVE":
            if semantic_plan is None:
                raise ValueError("Semantic plan required before approving equity mapping")
            category = semantic_by_target[target].get("targetKind")
            known = KNOWN.get(target)
            if known is None or known[0] != category or known[2] != (
                semantic_by_target[target].get("candidateRelatedListedParentSymbol") or ""
            ):
                raise ValueError("Typed research metadata does not match source catalog")
            if category != "LISTED_PARENT_CANDIDATE":
                raise ValueError(
                    f"Equity scoring not suitable for target measurement type: "
                    f"{target} ({category})"
                )
            approved_row = _validate_approval(row)
            expected_symbol = semantic_by_target[target].get(
                "candidateRelatedListedParentSymbol"
            )
            if (approved_row["mapping_kind"] != "DIRECT_EQUITY"
                    or approved_row["symbol"] != expected_symbol):
                raise ValueError("Documented parent symbol mismatch")
            approved.append(approved_row)
        else:
            if any(row[field].strip() for field in REVIEW_EDIT_FIELDS[1:]):
                raise ValueError(f"Unapproved target contains instrument fields: {target}")
            if decision == "PENDING":
                deferred.append(target)
            else:
                rejected.append(target)
    if reviewed != set(expected):
        raise ValueError("Incomplete review queue")

    # Copy original mapping data without rewriting earlier evidence or
    # replacing approved targets. Existing mappings have priority.
    original_rows = _csv(original_mappings, {"target", "market", "symbol", "benchmark",
                                              "verified_at"})
    _mappings(original_mappings)
    if len(original_rows) > 2000:
        raise ValueError("Original mapped cohort exceeds bounded size")
    mapping_rows = [{
        name: item.get(name, "") or "" for name in CSV_FIELDS
    } for item in original_rows]
    seen = {item["target"] for item in mapping_rows}
    if any(item["target"] in seen for item in approved):
        raise ValueError("Do not override an existing mapped target")
    mapping_rows.extend(approved)
    mapping_rows.sort(key=lambda item: item["target"])
    # Existing adjusted prices may be from different observed provider
    # snapshots, but do not mix prices within an existing instrument here.
    price_series = _observations(price_csv)
    present_symbols = set(price_series)
    added = {item["target"] for item in approved}
    needed = sorted({
        instrument
        for item in approved
        for instrument in (item["symbol"], item["benchmark"])
        if instrument not in present_symbols
    })
    to_collect = sorted([
        item for item in approved
        if item["symbol"] in needed or item["benchmark"] in needed
    ], key=lambda item: item["target"])
    return {
        "schemaVersion": "reviewed_market_mapping_plan_v1",
        "gapReportSha256": _hash(gap_report),
        "originalOutcomesSha256": _hash(source_outcomes),
        "reviewCSVSha256": _hash(reviewed_csv),
        "semanticPlanSha256": (
            _hash(semantic_plan) if semantic_plan is not None else None
        ),
        "economicTargetTypesGuarded": semantic_plan is not None,
        "originalMappingsSha256": _hash(original_mappings),
        "existingAdjustedPricesSha256": _hash(price_csv),
        "existingMappedTargets": len(original_rows),
        "newlyApprovedTargets": len(approved),
        "approvedNewTargets": sorted(added),
        "rejectedAsUnrepresentableTargets": sorted(rejected),
        "pendingTargets": sorted(deferred),
        "newPriceSymbolsWithoutAnyStoredDailyBars": needed,
        "newApprovedTargetsRequiringPriceCollection": len(to_collect),
        "historicalCoverageNotGuaranteed": True,
        "approvedBySourceDocumentationIndependently": False,
        "prospectivePredictionAccuracy": None,
        "trainingEligible": False,
        "noMappingsAutoApproved": True,
        "next": (
            "Only explicitly APPROVE reviewed targets are added. "
            "Fetch missing real adjusted daily prices for collection targets, "
            "recover intact new-symbol series with the existing recovery CLI, "
            "then rerun the retrospective evaluator. Never backdate source evidence."
        ),
    }, mapping_rows, to_collect


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply provenance-reviewed target mappings")
    parser.add_argument("--gap-report", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--mappings", type=Path, required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--semantic-plan", type=Path,
                        help="Required if approving any equity mappings")
    parser.add_argument("--output-mappings", type=Path, required=True)
    parser.add_argument("--output-price-queue", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    destinations = (args.output_mappings, args.output_price_queue, args.output_report)
    try:
        if (len({path.resolve() for path in destinations}) != len(destinations)
                or any(path.exists() or path.is_symlink() for path in destinations)):
            raise ValueError("Outputs must be distinct new files; prior results immutable")
        report, mapped, collect = apply_review(
            args.gap_report, args.outcomes, args.review_csv,
            args.mappings, args.prices, semantic_plan=args.semantic_plan,
        )
        with args.output_mappings.open("x", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(mapped)
        with args.output_price_queue.open("x", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(collect)
        with args.output_report.open("x", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(json.dumps({
            "newlyApprovedTargets": report["newlyApprovedTargets"],
            "pendingTargets": len(report["pendingTargets"]),
            "newPriceSymbolsWithoutAnyStoredDailyBars": (
                report["newPriceSymbolsWithoutAnyStoredDailyBars"]
            ),
            "mappingOutput": str(args.output_mappings),
            "priceCollectionMappingOutput": str(args.output_price_queue),
            "trainingEligible": False,
        }, ensure_ascii=False))
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.exit(2, f"reviewed-market-mapping: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
