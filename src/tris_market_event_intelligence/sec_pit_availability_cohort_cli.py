"""Offline SEC disclosure-availability evidence ledger and ex-ante review cohort.

SEC acceptance is a regulator receipt clock, not observed public delivery.
Retrospective post-event returns never enter feature review selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import UTC, date, datetime
from pathlib import Path

from tris_market_event_intelligence.historical_issuer_corroboration_cli import _write
from tris_market_event_intelligence.sec_coverage_reconciliation_cli import read_inventory

MAX_REPORT_BYTES = 1_500_000
MAX_CANDIDATES = 15000
MAX_RESEARCH_EVENTS = 500
SHA = re.compile(r"[0-9a-f]{64}\Z")
ACCESSION = re.compile(r"0001045810-\d{2}-\d{6}\Z")
FORMS = ("8-K", "10-Q", "10-K")
HORIZONS = ("1", "3", "5")


def _read_study(path: Path) -> tuple[dict, str]:
    if (path.is_symlink() or not path.is_file()
            or path.stat().st_size > MAX_REPORT_BYTES):
        raise ValueError("SEC_STUDY_FILE_INVALID")
    raw = path.read_bytes()
    if len(raw) > MAX_REPORT_BYTES:
        raise ValueError("SEC_STUDY_TOO_LARGE")
    study = json.loads(raw)
    if not isinstance(study, dict):
        raise TypeError("SEC_STUDY_OBJECT_REQUIRED")
    return study, hashlib.sha256(raw).hexdigest()


def _validated_study(study: dict) -> dict[str, dict]:
    if (study.get("schemaVersion") != "historical_sec_multiyear_price_study_v1"
            or study.get("status") !=
            "RAW_SEC_SOURCES_PINNED_RETROSPECTIVE_PRICE_OUTCOMES"
            or study.get("ticker") != "NVDA"
            or study.get("benchmark") != "SPY"
            or study.get("selectionRule") !=
            "EX_ANTE_FIRST_EIGHT_K_PER_YEAR_FROM_SEC_COVERAGE_REVIEW_QUEUE"
            or study.get("trainingReady") is not False
            or study.get("predictionReady") is not False
            or study.get("causalPriceImpactEstablished") is not False
            or study.get("newsAbsenceVerifiedForControls") is not False
            or study.get("firstPublisherPublicationTimeVerified") is not False
            or study.get("priceMovementBeforeDecisionExcluded") is not False
            or study.get("retrospectiveOutcomeAlreadyKnown") is not True
            or study.get("modelPromotionAllowed") is not False):
        raise ValueError("SEC_STUDY_PIT_PROVENANCE_INVALID")
    rows = study.get("events")
    if (not isinstance(rows, list)
            or not 1 <= len(rows) <= MAX_RESEARCH_EVENTS
            or len(rows) != study.get("eventCount")
            or not all(isinstance(x, dict) for x in rows)):
        raise ValueError("SEC_STUDY_EVENT_COUNT_INVALID")
    mapping: dict[str, dict] = {}
    raw_count = 0
    for row in rows:
        acc = row.get("accession")
        if not isinstance(acc, str) or not ACCESSION.fullmatch(acc) or acc in mapping:
            raise ValueError("SEC_STUDY_ACCESSION_INVALID")
        if (not isinstance(row.get("secIndexSha256"), str)
                or not SHA.fullmatch(row["secIndexSha256"])
                or row.get("issuerFilingIndexContentVerified") is not True
                or row.get("trainingEligible") is not False
                or row.get("sourceAvailableByDecisionCutoff") is not False
                or row.get("independentPublicAvailabilityAtCutoffVerified") is not False
                or row.get("publisherFirstReleaseTimeVerified") is not False
                or row.get("secReceiptBeforeDecisionCutoff") is not True
                or row.get("secReceiptTiming") not in (
                    "PREOPEN_SAME_SESSION", "AFTER_CLOSE_NEXT_SESSION"
                )):
            raise ValueError("SEC_STUDY_SOURCE_ATTESTATION_INVALID")
        accepted = datetime.fromisoformat(row["secAcceptedAt"])
        decision = datetime.fromisoformat(row["decisionAt"])
        if (accepted.utcoffset() is None or decision.utcoffset() is None
                or accepted.astimezone(UTC) >= decision.astimezone(UTC)
                or not isinstance(row.get("eventSession"), str)
                or date.fromisoformat(row["eventSession"]) != decision.date()
                and not (decision.astimezone(UTC).date()
                         == date.fromisoformat(row["eventSession"]))):
            # A UTC decision may be on the same NY day or its UTC date.
            raise ValueError("SEC_STUDY_DECISION_TIME_INVALID")
        has_exhibit = row.get("secExhibit99_1Sha256") is not None
        if (has_exhibit and (
            not isinstance(row["secExhibit99_1Sha256"], str)
            or not SHA.fullmatch(row["secExhibit99_1Sha256"])
            or row.get("issuerExhibitContentVerified") is not True
        )) or (not has_exhibit and row.get("issuerExhibitContentVerified") is not False):
            raise ValueError("SEC_STUDY_EXHIBIT_INVALID")
        outcomes = row.get("outcome")
        if (not isinstance(outcomes, dict)
                or outcomes.get("includesMovementBeforeDecision") is not True
                or outcomes.get("returnWindow") != "PREVIOUS_CLOSE_TO_TARGET_CLOSE"
                or not isinstance(outcomes.get("horizonsTradingSessions"), dict)
                or set(outcomes["horizonsTradingSessions"]) != set(HORIZONS)):
            raise ValueError("SEC_STUDY_RETROSPECTIVE_OUTCOME_INVALID")
        mapping[acc] = row
        raw_count += 1 + has_exhibit
    if raw_count != study.get("rawSourceCount"):
        raise ValueError("SEC_STUDY_RAW_SOURCE_COUNT_INVALID")
    return mapping


def _validate_inventory(inventory: dict, study: dict) -> list[dict]:
    if (inventory.get("schemaVersion") != "sec_historical_filing_inventory_v1"
            or inventory.get("status") !=
            "SEC_SUBMISSIONS_CANDIDATES_REQUIRE_INDEPENDENT_TIME_AND_EXHIBIT_VERIFICATION"
            or inventory.get("companyCik") != "0001045810"
            or inventory.get("ticker") != "NVDA"
            or inventory.get("trainingReady") is not False
            or inventory.get("originalFilingExhibitsPinned") is not False
            or inventory.get("firstPublisherPublicationTimeVerified") is not False
            or inventory.get("secAcceptanceTimeIndependentlyVerified") is not False):
        raise ValueError("SEC_INVENTORY_PIT_PROVENANCE_INVALID")
    rows = inventory.get("filings")
    if (not isinstance(rows, list) or not 1 <= len(rows) <= MAX_CANDIDATES
            or len(rows) != inventory.get("candidateFilings")
            or inventory.get("sourcePages") != len(inventory.get("sources", []))
            or study.get("inventorySha256") is None):
        raise ValueError("SEC_INVENTORY_CARDINALITY_INVALID")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("SEC_INVENTORY_ROW_INVALID")
        acc = row.get("accession")
        if (not isinstance(acc, str) or not ACCESSION.fullmatch(acc) or acc in seen
                or row.get("form") not in FORMS
                or not isinstance(row.get("filingDate"), str)
                or not isinstance(row.get("secIndexUrl"), str)
                or not row["secIndexUrl"].startswith(
                    "https://www.sec.gov/Archives/edgar/data/1045810/"
                )
                or row.get("eligibleAsNewsModelFeature") is not False):
            raise ValueError("SEC_INVENTORY_ROW_PROVENANCE_INVALID")
        date.fromisoformat(row["filingDate"])
        seen.add(acc)
    return rows


def build_sec_pit_cohort(
    *, inventory: dict, inventory_sha: str,
    study: dict, study_sha: str, max_per_year_form: int = 3,
) -> dict:
    """Choose calendar/form-stratified original SEC research candidates.

    Earnings outcomes are never consulted for cohort selection.
    """
    if (not isinstance(max_per_year_form, int) or isinstance(max_per_year_form, bool)
            or not 1 <= max_per_year_form <= 3
            or not SHA.fullmatch(inventory_sha)
            or not SHA.fullmatch(study_sha)):
        raise ValueError("SEC_PIT_COHORT_SETTINGS_INVALID")
    reviewed = _validated_study(study)
    filings = _validate_inventory(inventory, study)
    if inventory_sha != study["inventorySha256"]:
        raise ValueError("SEC_INVENTORY_SHA_NOT_STUDY_PINNED")
    stock_sha = study.get("stockSha256")
    benchmark_sha = study.get("benchmarkSha256")
    if not isinstance(stock_sha, str) or not SHA.fullmatch(stock_sha):
        raise ValueError("SEC_STUDY_STOCK_SHA_INVALID")
    if not isinstance(benchmark_sha, str) or not SHA.fullmatch(benchmark_sha):
        raise ValueError("SEC_STUDY_BENCHMARK_SHA_INVALID")
    by_accession = {row["accession"]: row for row in filings}
    if not set(reviewed) <= set(by_accession):
        raise ValueError("SEC_REVIEWED_EVENT_NOT_IN_INVENTORY")
    calendar_forms: dict[tuple[str, str], list[dict]] = defaultdict(list)
    statuses = []
    for row in sorted(filings, key=lambda r: (r["filingDate"], r["accession"])):
        accession = row["accession"]
        evidence = reviewed.get(accession)
        calendar_forms[(row["filingDate"][:4], row["form"])].append(row)
        statuses.append({
            "accession": accession,
            "form": row["form"],
            "filingDate": row["filingDate"],
            "secIndexUrl": row["secIndexUrl"],
            "evidenceStatus": (
                "SEC_RAW_INDEX_PINNED_PUBLICATION_CLOCK_UNKNOWN"
                if evidence else "SEC_SUBMISSIONS_METADATA_ONLY"
            ),
            "secIndexSha256": evidence["secIndexSha256"] if evidence else None,
            "secExhibit99_1Sha256": (
                evidence["secExhibit99_1Sha256"] if evidence else None
            ),
            "secReceiptAt": evidence["secAcceptedAt"] if evidence else None,
            "secReceiptMarketSession": evidence["eventSession"] if evidence else None,
            "secReceiptTiming": evidence["secReceiptTiming"] if evidence else None,
            "firstPublisherPublishedAt": None,
            "firstSecPubliclyDisseminatedAt": None,
            "firstIndependentObservedAt": None,
            "marketAvailableAt": None,
            "publicAvailabilityVerifiedByCutoff": False,
            "usableAsPreopenModelFeature": False,
            "absenceOfOtherNewsVerified": False,
            "canBeUsedAsNoNewsControl": False,
            "retrospectiveOutcomeIsTrainingLabel": False,
        })
    selected: list[dict] = []
    for (year, form), rows in sorted(calendar_forms.items()):
        # Exact start/centre/end are deterministic based only on
        # SEC filing date/accession, never future price outcomes.
        ordered = sorted(rows, key=lambda row: (row["filingDate"], row["accession"]))
        positions = (
            [0] if max_per_year_form == 1 else
            [0, len(ordered) - 1] if max_per_year_form == 2 else
            [0, (len(ordered) - 1) // 2, len(ordered) - 1]
        )
        for position in dict.fromkeys(positions):
            item = ordered[position]
            selected.append({
                "accession": item["accession"],
                "form": form,
                "year": year,
                "filingDate": item["filingDate"],
                "secIndexUrl": item["secIndexUrl"],
                "alreadyRawPinned": item["accession"] in reviewed,
                "researchTask": (
                    "INDEPENDENT_PUBLIC_TIME_EVIDENCE"
                    if item["accession"] in reviewed else
                    "ORIGINAL_SEC_INDEX_EXHIBIT_AND_PUBLIC_TIME_EVIDENCE"
                ),
                "selectedWithoutFutureReturns": True,
            })
    return {
        "schemaVersion": "sec_pit_availability_research_cohort_v1",
        "status": "SEC_PUBLIC_AVAILABILITY_NOT_INDEPENDENTLY_OBSERVED",
        "ticker": "NVDA",
        "benchmark": "SPY",
        "inventorySha256": inventory_sha,
        "multiyearStudySha256": study_sha,
        "stockSha256": stock_sha,
        "benchmarkSha256": benchmark_sha,
        "filingCandidates": len(filings),
        "secRawPinnedCases": len(reviewed),
        "metadataOnlyCases": len(filings) - len(reviewed),
        "knownFirstPublicAvailabilityCases": 0,
        "unverifiedNewsStatusDaysAreNegativeControls": False,
        "selectionRule": "CALENDAR_YEAR_FORM_FIRST_MIDDLE_LAST_NO_PRICE_RANKING",
        "maxPerYearForm": max_per_year_form,
        "reviewQueueCount": len(selected),
        "reviewQueue": selected,
        "filingAvailabilityLedger": statuses,
        "disseminationClockIndependentSourceRequired": True,
        "secReceiptClockIsNotPublicAvailability": True,
        "priceOutcomesUsedForSelection": False,
        "previousCloseToFutureCloseNotExecutableOpenEntry": True,
        "noClaimsOfPreopenSourceAvailability": True,
        "trainingReady": False,
        "backtestReady": False,
        "predictionReady": False,
        "modelPromotionAllowed": False,
        "finalTestTouched": False,
        "productionDbModified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PIT-safe SEC publication gaps and date/form-stratified research"
    )
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--max-per-year-form", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    try:
        if not args.preview and args.output is None:
            raise ValueError("SEC_PIT_COHORT_OUTPUT_REQUIRED")
        inventory, inventory_sha = read_inventory(args.inventory)
        study, study_sha = _read_study(args.study)
        result = build_sec_pit_cohort(
            inventory=inventory, inventory_sha=inventory_sha,
            study=study, study_sha=study_sha,
            max_per_year_form=args.max_per_year_form,
        )
        if not args.preview:
            _write(args.output, result)
        print(json.dumps({
            "status": result["status"],
            "filingCandidates": result["filingCandidates"],
            "secRawPinnedCases": result["secRawPinnedCases"],
            "metadataOnlyCases": result["metadataOnlyCases"],
            "knownFirstPublicAvailabilityCases": 0,
            "reviewQueueCount": result["reviewQueueCount"],
            "trainingReady": False,
            "outputCreated": not args.preview,
        }))
    except (OSError, ValueError, TypeError, KeyError, UnicodeError,
            json.JSONDecodeError):
        parser.exit(2, "sec-pit-cohort: SOURCE_PROVENANCE_OR_PIT_INVALID\n")


if __name__ == "__main__":
    main()
