"""Explicit opt-in SEC original-index source custody for precommitted PIT queue.

A post-hoc download is proof of ORIGINAL BYTES AT CAPTURE, not proof of
historical first public availability. No event is made training-eligible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit

import httpx

from tris_market_event_intelligence.historical_issuer_corroboration_cli import _write
from tris_market_event_intelligence.sec_multiyear_price_study_cli import _html

COHORT_SCHEMA = "sec_pit_availability_research_cohort_v1"
RESULT_SCHEMA = "sec_pit_original_index_capture_v1"
SOURCE_SIZE = 1_500_000
COHORT_SIZE = 2_500_000
MAX_LIMIT = 69
SHA = re.compile(r"[a-f0-9]{64}\Z")
ACC = re.compile(r"\d{10}-\d{2}-\d{6}\Z")
SEC_ACCEPTED = re.compile(r"Accepted\s+(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})", re.I)


def _read_cohort(path: Path) -> tuple[dict, str]:
    if (path.is_symlink() or not path.is_file()
            or path.stat().st_size > COHORT_SIZE):
        raise ValueError("SEC_COHORT_SOURCE_INVALID")
    raw = path.read_bytes()
    if len(raw) > COHORT_SIZE:
        raise ValueError("SEC_COHORT_SOURCE_TOO_LARGE")
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise TypeError("SEC_COHORT_OBJECT_REQUIRED")
    return result, hashlib.sha256(raw).hexdigest()


def _url_allowed(url: object, accession: str) -> bool:
    if not isinstance(url, str) or not ACC.fullmatch(accession):
        return False
    part = urlsplit(url)
    expected = (
        "/Archives/edgar/data/1045810/"
        + accession.replace("-", "")
        + "/"
        + accession
        + "-index.htm"
    )
    return (part.scheme == "https" and part.netloc == "www.sec.gov"
            and part.path == expected and part.query == ""
            and part.fragment == "" and unquote(part.path) == part.path)


def _validate_cohort(cohort: dict) -> list[dict]:
    if (cohort.get("schemaVersion") != COHORT_SCHEMA
            or cohort.get("status") !=
            "SEC_PUBLIC_AVAILABILITY_NOT_INDEPENDENTLY_OBSERVED"
            or cohort.get("ticker") != "NVDA"
            or cohort.get("benchmark") != "SPY"
            or cohort.get("selectionRule") !=
            "CALENDAR_YEAR_FORM_FIRST_MIDDLE_LAST_NO_PRICE_RANKING"
            or cohort.get("priceOutcomesUsedForSelection") is not False
            or cohort.get("secReceiptClockIsNotPublicAvailability") is not True
            or cohort.get("trainingReady") is not False
            or cohort.get("predictionReady") is not False
            or cohort.get("modelPromotionAllowed") is not False
            or cohort.get("knownFirstPublicAvailabilityCases") != 0):
        raise ValueError("SEC_COHORT_PROVENANCE_INVALID")
    queue = cohort.get("reviewQueue")
    ledger = cohort.get("filingAvailabilityLedger")
    if (not isinstance(queue, list) or not isinstance(ledger, list)
            or not 1 <= len(queue) <= MAX_LIMIT
            or len(queue) != cohort.get("reviewQueueCount")
            or len(ledger) != cohort.get("filingCandidates")):
        raise ValueError("SEC_COHORT_CARDINALITY_INVALID")
    by_id = {}
    for record in ledger:
        if not isinstance(record, dict):
            raise TypeError("SEC_LEDGER_RECORD_INVALID")
        accession = record.get("accession")
        if not isinstance(accession, str) or not ACC.fullmatch(accession) or accession in by_id:
            raise ValueError("SEC_LEDGER_ACCESSION_INVALID")
        if (record.get("publicAvailabilityVerifiedByCutoff") is not False
                or record.get("usableAsPreopenModelFeature") is not False
                or record.get("canBeUsedAsNoNewsControl") is not False):
            raise ValueError("SEC_LEDGER_UNAUTHORIZED_PROMOTION")
        by_id[accession] = record
    selected: set[str] = set()
    for task in queue:
        if not isinstance(task, dict):
            raise TypeError("SEC_COHORT_TASK_INVALID")
        accession = task.get("accession")
        if (not isinstance(accession, str) or not ACC.fullmatch(accession)
                or accession in selected
                or accession not in by_id
                or task.get("form") not in ("8-K", "10-K", "10-Q")
                or task.get("form") != by_id[accession].get("form")
                or task.get("filingDate") != by_id[accession].get("filingDate")
                or task.get("secIndexUrl") != by_id[accession].get("secIndexUrl")
                or not _url_allowed(task.get("secIndexUrl"), accession)
                or task.get("selectedWithoutFutureReturns") is not True):
            raise ValueError("SEC_COHORT_TASK_PROVENANCE_INVALID")
        pinned = by_id[accession].get("secIndexSha256")
        if ((pinned is not None and (not isinstance(pinned, str)
                                     or not SHA.fullmatch(pinned)))
                or task.get("alreadyRawPinned") is not (pinned is not None)):
            raise ValueError("SEC_COHORT_PIN_MISMATCH")
        selected.add(accession)
    return queue


def _verify_index(raw: bytes, task: dict) -> str:
    doc = _html(raw)
    text = " ".join(doc.words)
    collapsed = re.sub(r"\s+", " ", text)
    accession = task["accession"]
    if (accession not in collapsed
            or not re.search(r"\bForm\s+" + re.escape(task["form"]) + r"\b",
                             collapsed, re.IGNORECASE)
            or "NVIDIA" not in collapsed.upper()
            or not re.search(r"Filing Date\s+" + re.escape(task["filingDate"]),
                             collapsed, re.IGNORECASE)):
        raise ValueError("SEC_RAW_FILING_METADATA_MISMATCH")
    match = SEC_ACCEPTED.search(collapsed)
    if not match:
        raise ValueError("SEC_RAW_ACCEPTED_HEADER_MISSING")
    accepted = f"{match.group(1)}T{match.group(2)}"
    datetime.fromisoformat(accepted)
    return accepted


def _fetch(client: httpx.Client, url: str, user_agent: str) -> bytes:
    with client.stream("GET", url, follow_redirects=False, headers={
        "User-Agent": user_agent,
        "Accept": "text/html",
    }) as response:
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError("SEC_SOURCE_NOT_SUCCESS")
        raw = bytearray()
        for chunk in response.iter_bytes():
            raw.extend(chunk)
            if len(raw) > SOURCE_SIZE:
                raise ValueError("SEC_SOURCE_OVERSIZED")
    return bytes(raw)


def capture_sec_pit_evidence(
    *, cohort: dict, cohort_sha: str, source_dir: Path,
    limit: int, client: httpx.Client | None = None,
    user_agent: str | None = None, persist_raw: bool = False,
) -> dict:
    tasks = _validate_cohort(cohort)
    if (not isinstance(cohort_sha, str) or not SHA.fullmatch(cohort_sha)
            or isinstance(limit, bool) or not isinstance(limit, int)
            or not 1 <= limit <= MAX_LIMIT
            or not source_dir.is_dir() or source_dir.is_symlink()):
        raise ValueError("SEC_CAPTURE_SETTINGS_INVALID")
    by_id = {entry["accession"]: entry for entry in cohort["filingAvailabilityLedger"]}
    selected = tasks[:limit]
    rows = []
    downloads = 0
    for task in selected:
        acc = task["accession"]
        local = source_dir / f"sec-{acc}-index.html"
        expected_sha = by_id[acc].get("secIndexSha256")
        if local.is_symlink():
            raise ValueError("SEC_SOURCE_SYMLINK_REJECTED")
        if local.exists():
            if not local.is_file() or local.stat().st_size > SOURCE_SIZE:
                raise ValueError("SEC_SOURCE_LOCAL_INVALID")
            raw = local.read_bytes()
            already_cached = True
        else:
            if client is None or user_agent is None:
                raise ValueError("SEC_SOURCE_NOT_YET_CAPTURED")
            if downloads:
                time.sleep(0.55)
            raw = _fetch(client, task["secIndexUrl"], user_agent)
            already_cached = False
            downloads += 1
        accepted_header = _verify_index(raw, task)
        source_sha = hashlib.sha256(raw).hexdigest()
        if expected_sha is not None and source_sha != expected_sha:
            raise ValueError("SEC_SOURCE_SHA_MISMATCH")
        if persist_raw and not already_cached:
            from tris_market_event_intelligence.historical_source_acquisition import (
                create_private_artifact,
            )
            create_private_artifact(local, raw)
        rows.append({
            "accession": acc,
            "form": task["form"],
            "filingDate": task["filingDate"],
            "secIndexUrl": task["secIndexUrl"],
            "secIndexSha256": source_sha,
            "secAcceptedLocalUnverifiedTimezone": accepted_header,
            "secReceiptClockOnly": True,
            "originalIndexAuthenticatedByContentAndSecHost": True,
            "fetchedInThisRun": not already_cached,
            "historicalFirstPublicAvailabilityAt": None,
            "firstIndependentHistoricalObservedAt": None,
            "hasIndependentHistoricalPublicationClockEvidence": False,
            "verifiedPreopenMarketAvailability": False,
            "eligibleForNewsModelTraining": False,
        })
    return {
        "schemaVersion": RESULT_SCHEMA,
        "status": "SEC_ORIGINAL_INDEX_SHA_PINNED_PUBLICATION_TIME_UNPROVEN",
        "cohortSha256": cohort_sha,
        "ticker": "NVDA",
        "selectedTaskCount": len(selected),
        "capturedIndexCount": len(rows),
        "previouslyCachedCount": len(rows) - downloads,
        "newOriginalIndexCount": downloads,
        "sourceRecords": rows,
        "secAcceptanceIsNotFirstPublicRelease": True,
        "secPublicDisseminationLagUnboundedHistorically": True,
        "knownFirstPublicAvailabilityCases": 0,
        "trainingReady": False,
        "backtestReady": False,
        "causalityVerified": False,
        "modelPromotionAllowed": False,
        "productionDbModified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SEC original-index source PIN evidence")
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, choices=range(1, MAX_LIMIT + 1), default=12)
    parser.add_argument("--output", type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preview", action="store_true")
    modes.add_argument("--run", action="store_true")
    args = parser.parse_args()
    try:
        cohort, digest = _read_cohort(args.cohort)
        if args.run and (args.output is None or args.output.exists()
                         or args.output.is_symlink()):
            raise ValueError("SEC_CAPTURE_NEW_OUTPUT_REQUIRED")
        user_agent = os.environ.get("TRIS_SEC_USER_AGENT", "")
        if args.run and (not 12 <= len(user_agent) <= 200
                         or "@" not in user_agent
                         or "\r" in user_agent or "\n" in user_agent):
            raise ValueError("SEC_USER_AGENT_INVALID")
        with httpx.Client(timeout=httpx.Timeout(30), trust_env=False) as client:
            report = capture_sec_pit_evidence(
                cohort=cohort, cohort_sha=digest,
                source_dir=args.source_dir, limit=args.limit,
                client=client if args.run else None,
                user_agent=user_agent if args.run else None,
                persist_raw=args.run,
            )
        if args.run:
            _write(args.output, report)
        print(json.dumps({
            "status": report["status"],
            "selected": report["selectedTaskCount"],
            "originalIndexesPinned": report["capturedIndexCount"],
            "cached": report["previouslyCachedCount"],
            "newIndexes": report["newOriginalIndexCount"],
            "firstPublicTimesVerified": 0,
            "trainingReady": False,
            "outputCreated": args.run,
        }))
    except httpx.HTTPStatusError as exc:
        parser.exit(2, f"sec-publication-capture: SEC_HTTP_{exc.response.status_code}\n")
    except httpx.HTTPError:
        parser.exit(2, "sec-publication-capture: SEC_NETWORK_FAILURE\n")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, UnicodeError):
        parser.exit(2, "sec-publication-capture: SOURCE_OR_PIT_EVIDENCE_INVALID\n")


if __name__ == "__main__":
    main()
