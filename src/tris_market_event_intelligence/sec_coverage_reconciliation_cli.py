"""Reconcile SEC filing *candidate metadata* to an existing PIT-safe market universe.

This is coverage triage, not retrospective event labelling or news-negative
controls. SEC submission metadata has no independently checked market-time
availability; a filing date may correspond to multiple possible trade days.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from bisect import bisect_left, bisect_right
from collections import Counter
from datetime import date
from pathlib import Path

from tris_market_event_intelligence.historical_issuer_corroboration_cli import _write
from tris_market_event_intelligence.historical_sec_event_universe_cli import (
    build_sec_event_universe,
)

MAX_INVENTORY_BYTES = 8_000_000
MAX_CANDIDATES = 15_000
ACCESSION = re.compile(r"^\d{10}-\d{2}-\d{6}$")
HEX64 = re.compile(r"^[a-f0-9]{64}$")


def read_inventory(path: Path) -> tuple[dict, str]:
    if (path.is_symlink() or not path.is_file()
            or path.stat().st_size > MAX_INVENTORY_BYTES):
        raise ValueError("SEC_INVENTORY_FILE_INVALID")
    raw = path.read_bytes()
    if len(raw) > MAX_INVENTORY_BYTES:
        raise ValueError("SEC_INVENTORY_TOO_LARGE")
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise TypeError("SEC_INVENTORY_OBJECT_REQUIRED")
    return result, hashlib.sha256(raw).hexdigest()


def reconcile_sec_coverage(*, inventory: dict, inventory_sha: str,
                           universe: dict, max_candidates: int = 200) -> dict:
    if (not 1 <= max_candidates <= 1000 or not HEX64.fullmatch(inventory_sha)
            or inventory.get("schemaVersion") != "sec_historical_filing_inventory_v1"
            or inventory.get("status") !=
            "SEC_SUBMISSIONS_CANDIDATES_REQUIRE_INDEPENDENT_TIME_AND_EXHIBIT_VERIFICATION"
            or inventory.get("companyCik") != "0001045810"
            or inventory.get("ticker") != universe.get("ticker")
            or inventory.get("secAcceptanceTimeIndependentlyVerified") is not False
            or inventory.get("originalFilingExhibitsPinned") is not False
            or inventory.get("firstPublisherPublicationTimeVerified") is not False
            or inventory.get("trainingReady") is not False
            or universe.get("schemaVersion") != "historical_sec_event_universe_v1"
            or universe.get("status") != "MULTI_SESSION_RETROSPECTIVE_SEC_EVIDENCE"
            or universe.get("nonEventSessionsAreTrueNoNewsControls") is not False
            or universe.get("trainingReady") is not False
            or universe.get("evidenceCoverageComplete") is not False):
        raise ValueError("SEC_COVERAGE_PROVENANCE_MISMATCH")
    sources = inventory.get("sources")
    if (not isinstance(sources, list) or not sources
            or len(sources) != inventory.get("sourcePages")
            or any(not isinstance(source, dict)
                   or not isinstance(source.get("name"), str)
                   or not HEX64.fullmatch(str(source.get("sha256")))
                   for source in sources)):
        raise ValueError("SEC_INVENTORY_SOURCE_LINEAGE_INVALID")
    filings, records = inventory.get("filings"), universe.get("records")
    if (not isinstance(filings, list) or not isinstance(records, list)
            or len(filings) > MAX_CANDIDATES or not records
            or len(filings) != inventory.get("candidateFilings")
            or len(records) != universe.get("sessionCount")):
        raise ValueError("SEC_COVERAGE_CARDINALITY_MISMATCH")
    by_session = {}
    pinned = {}
    for record in records:
        day = record.get("session")
        features = record.get("features")
        if not isinstance(day, str) or not isinstance(features, dict) or day in by_session:
            raise ValueError("SEC_UNIVERSE_SESSION_INVALID")
        if record.get("trainingEligible") is not False:
            raise ValueError("SEC_UNIVERSE_TRAINING_PROMOTION")
        by_session[day] = record
        events = features.get("pinnedSecEvents")
        if not isinstance(events, list) or len(events) > 1:
            raise ValueError("SEC_UNIVERSE_EVENT_INVALID")
        for event in events:
            acc = event.get("secAccession")
            if (not isinstance(acc, str) or not ACCESSION.fullmatch(acc)
                    or acc in pinned):
                raise ValueError("SEC_UNIVERSE_ACCESSION_INVALID")
            pinned[acc] = day
    if (len(pinned) != universe.get("eventCount")
            or len(pinned) != universe.get("sessionsWithPinnedSecEvent")):
        raise ValueError("SEC_PINNED_EVENT_COUNT_MISMATCH")
    sessions = sorted(by_session)
    # Calendar-day ambiguity is deliberate: same-day and first subsequent
    # session are possible review targets; neither is a verified SEC cutoff.
    seen: set[str] = set()
    queue: list[dict] = []
    covered = []
    mapped_count = 0
    unmapped_count = 0
    by_form = Counter()
    for filing in filings:
        if not isinstance(filing, dict):
            raise TypeError("SEC_FILING_CANDIDATE_INVALID")
        acc = filing.get("accession")
        form = filing.get("form")
        filing_date = filing.get("filingDate")
        if isinstance(acc, str) and acc in seen:
            raise ValueError("SEC_FILING_CANDIDATES_NOT_UNIQUE")
        if (not isinstance(acc, str) or not ACCESSION.fullmatch(acc)
                or form not in ("8-K", "10-Q", "10-K")
                or not isinstance(filing_date, str)
                or filing.get("secAcceptanceTimeIndependentlyVerified") is not False
                or filing.get("filingExhibitFactsVerified") is not False
                or filing.get("eligibleAsNewsModelFeature") is not False
                or not isinstance(filing.get("rawSecAcceptanceDateTime"), str)
                or not isinstance(filing.get("secIndexUrl"), str)
                or not filing["secIndexUrl"].startswith(
                    "https://www.sec.gov/Archives/edgar/data/1045810/"
                )):
            raise ValueError("SEC_FILING_CANDIDATE_CONTRACT_INVALID")
        date.fromisoformat(filing_date)
        seen.add(acc)
        by_form[form] += 1
        if acc in pinned:
            covered.append({
                "accession": acc,
                "filingDate": filing_date,
                "verifiedEventSession": pinned[acc],
                "evidenceStatus": "INDEPENDENTLY_PINNED_SEC_EVENT",
            })
            continue
        left = bisect_left(sessions, filing_date)
        right = bisect_right(sessions, filing_date)
        possibilities = []
        if left < len(sessions) and sessions[left] == filing_date:
            possibilities.append(sessions[left])
        if right < len(sessions):
            possibilities.append(sessions[right])
        possibilities = list(dict.fromkeys(possibilities))
        if possibilities:
            mapped_count += 1
        else:
            unmapped_count += 1
        queue.append({
            "accession": acc,
            "form": form,
            "filingDate": filing_date,
            "secIndexUrl": filing["secIndexUrl"],
            "candidateSessions": possibilities,
            "reviewStatus": "METADATA_ONLY_VERIFY_SEC_ACCEPTANCE_AND_EXHIBIT",
            "rawAcceptanceClockTrustedForTrading": False,
            "candidateSessionsAreActualEventDates": False,
            "issuerFactsVerified": False,
            "eligibleForPredictionFeatures": False,
        })
    if len(seen) != inventory.get("candidateFilings"):
        raise ValueError("SEC_FILING_CANDIDATES_NOT_UNIQUE")
    # Missing candidate accession for an independently pinned packet does not
    # revoke that packet; the inventory may cover a narrower time period.
    pinned_not_in_inventory = sorted(set(pinned) - seen)
    queue.sort(key=lambda item: (item["filingDate"], item["accession"]))
    return {
        "schemaVersion": "historical_sec_coverage_reconciliation_v1",
        "status": "SEC_CANDIDATE_EVIDENCE_GAPS_REVIEW_REQUIRED",
        "ticker": universe["ticker"],
        "benchmark": universe["benchmark"],
        "inventorySha256": inventory_sha,
        "stockSha256": universe["stockSha256"],
        "benchmarkSha256": universe["benchmarkSha256"],
        "inventoryCandidateCount": len(filings),
        "inventoryFormCounts": dict(sorted(by_form.items())),
        "independentlyPinnedEventCount": len(pinned),
        "pinnedAccessionMatchesInInventory": len(covered),
        "pinnedAccessionsOutsideInventory": pinned_not_in_inventory,
        "unverifiedCandidateCount": len(queue),
        "unverifiedCandidatesWithSessionHints": mapped_count,
        "unmappedCandidates": unmapped_count,
        "reviewQueueTruncated": len(queue) > max_candidates,
        "reviewQueueLimit": max_candidates,
        "reviewQueue": queue[:max_candidates],
        "verifiedWithinInventory": covered,
        "marketSessionCount": universe["sessionCount"],
        "unknownNewsStatusSessionCount": universe["sessionsWithoutPinnedSecEvent"],
        "noCandidateDoesNotMeanNoNews": True,
        "filingDateIsNotPublisherOrMarketAvailability": True,
        "rawSecSubmissionClockNotIndependentlyVerified": True,
        "candidateWindowIsOnlyReviewHint": True,
        "trainingReady": False,
        "backtestReady": False,
        "finalTestTouched": False,
        "modelCalls": 0,
        "productionDbModified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline SEC submissions-to-price-universe coverage")
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--stock-csv", type=Path, required=True)
    parser.add_argument("--benchmark-csv", type=Path, required=True)
    parser.add_argument("--event", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    try:
        if not args.preview and args.output is None:
            raise ValueError("SEC_OUTPUT_REQUIRED")
        inventory, sha = read_inventory(args.inventory)
        universe = build_sec_event_universe(
            stock_csv=args.stock_csv, benchmark_csv=args.benchmark_csv,
            event_paths=args.event,
        )
        report = reconcile_sec_coverage(
            inventory=inventory, inventory_sha=sha, universe=universe,
            max_candidates=args.limit,
        )
        if not args.preview:
            _write(args.output, report)
        print(json.dumps({
            "status": report["status"],
            "pinnedEvents": report["independentlyPinnedEventCount"],
            "inventoryCandidates": report["inventoryCandidateCount"],
            "unverifiedCandidates": report["unverifiedCandidateCount"],
            "pinnedMatches": report["pinnedAccessionMatchesInInventory"],
            "unknownNewsStatusSessions": report["unknownNewsStatusSessionCount"],
            "reviewQueueTruncated": report["reviewQueueTruncated"],
            "trainingReady": False,
            "outputCreated": not args.preview,
        }))
    except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError):
        parser.exit(2, "sec-coverage-reconcile: INVALID_OR_UNVERIFIED_RESEARCH_INPUT\n")


if __name__ == "__main__":
    main()
