"""Read-only review of real Gemma direction/price evaluation coverage gaps.

Produces an actionable review queue without hallucinating a ticker,
selecting a sector ETF for a macro event, or increasing apparent accuracy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ALLOWED = {"UNMAPPED", "COMPLETED", "PARTIAL", "NEWS_SOURCE_NOT_ADMITTED"}
QUEUE_COLUMNS = (
    "target", "impacts", "articles", "article_ids", "predictions",
    "issue", "review_decision", "reviewed_market", "reviewed_symbol",
    "reviewed_benchmark", "mapping_kind", "mapping_evidence_url",
    "mapping_verified_at", "proxy_limitation",
)
MAX_BYTES = 20 * 1024 * 1024
MAX_IMPACTS = 20000


def analyze_gaps(report_path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    if (not report_path.is_file() or report_path.is_symlink()
            or report_path.stat().st_size > MAX_BYTES):
        raise ValueError("Expected existing bounded real-outcome JSON")
    payload = report_path.read_bytes()
    report = json.loads(payload)
    if not isinstance(report, dict) or report.get("schemaVersion") != (
        "market_outcome_validation_v1"
    ):
        raise ValueError("Unsupported historical market-outcome report")
    if report.get("researchOnly") is not True or report.get(
        "prospectivePredictionAccuracy"
    ) is not None or report.get("trainingEligible") is not False:
        raise ValueError("Historical results must retain unsealed research boundaries")
    impacts = report.get("items")
    if not isinstance(impacts, list) or len(impacts) > MAX_IMPACTS or (
        report.get("impactsInspected") != len(impacts)
    ):
        raise ValueError("Missing/partial impact coverage")
    if any(not isinstance(row, dict) for row in impacts):
        raise ValueError("Invalid impact record")

    status_counts: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    grouped: dict[str, list[dict]] = defaultdict(list)
    distinct_articles: dict[str, set[int]] = defaultdict(set)
    classified = []
    for impact in impacts:
        status = impact.get("status")
        if not isinstance(status, str):
            raise ValueError("Missing impact status")
        if status not in ALLOWED and status not in {
            "NO_ENTRY_SESSION", "BENCHMARK_ENTRY_MISSING",
        }:
            raise ValueError("Unexpected outcome status")
        if (impact.get("predictionSealedBeforeOutcome") is not False
                or impact.get("evaluationClass") not in {
                    "RETROSPECTIVE_UNSEALED", "RETROSPECTIVE_POST_EVENT_MAPPING",
                }):
            raise ValueError("Never accept prospective claims from historical outputs")
        article_id = impact.get("articleId")
        target = impact.get("target")
        if (type(article_id) is not int or article_id <= 0
                or not isinstance(target, str) or not target.strip()):
            raise ValueError("Missing article identity or target")
        gate = impact.get("sourceAuditStatus", "SOURCE_DB_NOT_REQUESTED")
        if not isinstance(gate, str):
            raise ValueError("Invalid source status")
        status_counts[status] += 1
        sources[gate] += 1
        if status == "NEWS_SOURCE_NOT_ADMITTED":
            issue = "NEWS_SOURCE_BLOCKED"
        elif status == "UNMAPPED":
            issue = "MAPPING_REQUIRED"
            grouped[target].append(impact)
            distinct_articles[target].add(article_id)
        elif status in {"NO_ENTRY_SESSION", "BENCHMARK_ENTRY_MISSING", "PARTIAL"}:
            issue = "INSUFFICIENT_MARKET_BARS"
        else:
            issue = "SOURCED_DAILY_OBSERVATION"
        classified.append({
            "articleId": article_id, "target": target, "status": status,
            "sourceAuditStatus": gate, "issue": issue,
            "prediction": impact.get("prediction"),
        })

    queue = []
    for target, entries in grouped.items():
        ids = sorted(distinct_articles[target])
        directions = Counter(str(x["prediction"]) for x in entries)
        queue.append({
            "target": target, "impacts": len(entries), "articles": len(ids),
            "article_ids": ";".join(str(i) for i in ids),
            "predictions": ";".join(
                f"{key}:{value}" for key, value in sorted(directions.items())
            ),
            "issue": "MAPPING_REQUIRED",
            "review_decision": "PENDING",
            "reviewed_market": "", "reviewed_symbol": "",
            "reviewed_benchmark": "", "mapping_kind": "",
            "mapping_evidence_url": "", "mapping_verified_at": "",
            "proxy_limitation": "",
        })
    queue.sort(key=lambda row: (-row["impacts"], row["target"]))
    counts = Counter(row["issue"] for row in classified)
    observed = len(impacts) - counts["MAPPING_REQUIRED"] - counts[
        "NEWS_SOURCE_BLOCKED"
    ] - counts["INSUFFICIENT_MARKET_BARS"]
    if observed < 0:
        raise ValueError("Internal classification inconsistency")
    summary = {
        "schemaVersion": "market_outcome_gap_review_v1",
        "inputSha256": hashlib.sha256(payload).hexdigest(),
        "sourceReportSchema": report["schemaVersion"],
        "totalImpacts": len(impacts),
        "statusCounts": dict(sorted(status_counts.items())),
        "sourceStatusCounts": dict(sorted(sources.items())),
        "reasonCounts": dict(sorted(counts.items())),
        "distinctUnmappedTargets": len(queue),
        "unmappedImpacts": counts["MAPPING_REQUIRED"],
        "blockedSourceImpacts": counts["NEWS_SOURCE_BLOCKED"],
        "priceBarGapImpacts": counts["INSUFFICIENT_MARKET_BARS"],
        "existingDailyObservedImpacts": observed,
        "maxSourceAdmissibleImpactsBeforeAdditionalSourceReview": (
            len(impacts) - counts["NEWS_SOURCE_BLOCKED"]
        ),
        "mappingQueue": queue,
        "impacts": classified,
        "reviewRequiredBeforeMappingApplied": True,
        "mappingApprovedAutomatically": False,
        "newPricesCreated": False,
        "trainingEligible": False,
        "prospectivePredictionAccuracy": None,
        "interpretation": (
            "A review queue is not a verified ticker map. "
            "Unmapped economic indicators, mixed sectors and nonlisted targets "
            "may legitimately remain unscored. Price outcomes are retrospective."
        ),
    }
    return summary, queue


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a source-preserving reviewed-mapping queue from real market results",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.resolve() == args.review_csv.resolve():
            raise ValueError("JSON and CSV destinations must differ")
        if any(path.exists() or path.is_symlink() for path in (
            args.output, args.review_csv,
        )):
            raise ValueError("Never overwrite source or prior review outputs")
        result, queue = analyze_gaps(args.report)
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write("\n")
        with args.review_csv.open("x", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=QUEUE_COLUMNS)
            writer.writeheader()
            writer.writerows(queue)
        print(json.dumps({
            "totalImpacts": result["totalImpacts"],
            "reasonCounts": result["reasonCounts"],
            "distinctUnmappedTargets": result["distinctUnmappedTargets"],
            "reviewCSV": str(args.review_csv),
            "report": str(args.output),
            "mappingApprovedAutomatically": False,
        }, ensure_ascii=False))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"market-outcome-gap-review: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
