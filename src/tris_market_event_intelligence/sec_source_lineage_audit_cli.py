"""Auditable overlap and historic first-observation evidence triage for SEC indexes.

Read only pre-existing, SHA-pinned private studies and capture reports.
Historical publication timestamps are never inferred from SEC acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from tris_market_event_intelligence.historical_issuer_corroboration_cli import _write
from tris_market_event_intelligence.sec_pit_availability_cohort_cli import _read_study
from tris_market_event_intelligence.sec_publication_evidence_capture_cli import (
    COHORT_SCHEMA,
    RESULT_SCHEMA,
    _read_cohort,
    _validate_cohort,
    _verify_index,
)

MAX_REPORT_BYTES = 2_500_000
MAX_WITNESS_BYTES = 300_000
ACC = re.compile(r"\d{10}-\d{2}-\d{6}\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")
SOURCE_TYPES = frozenset({"INDEPENDENT_MEDIA_ARCHIVE", "PUBLIC_FEED_ARCHIVE"})


def _read_json(path: Path, limit: int) -> tuple[dict, str]:
    if (path.is_symlink() or not path.is_file()
            or path.stat().st_size > limit):
        raise ValueError("SEC_LINEAGE_INPUT_FILE_INVALID")
    raw = path.read_bytes()
    if len(raw) > limit:
        raise ValueError("SEC_LINEAGE_INPUT_TOO_LARGE")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise TypeError("SEC_LINEAGE_INPUT_OBJECT_REQUIRED")
    return data, hashlib.sha256(raw).hexdigest()


def _valid_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("SEC_WITNESS_TIMESTAMP_REQUIRED")
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() is None:
        raise ValueError("SEC_WITNESS_TIMESTAMP_TIMEZONE_REQUIRED")
    return parsed.astimezone(UTC)


def _witness_records(
    witnesses: dict | None,
    *,
    source_dir: Path,
    allowed: dict[str, dict],
) -> dict[str, list[dict]]:
    if witnesses is None:
        return {}
    if (witnesses.get("schemaVersion") !=
            "sec_historical_first_observation_witnesses_v1"
            or witnesses.get("trainingReady") is not False
            or witnesses.get("independentAttestationComplete") is not False):
        raise ValueError("SEC_WITNESS_MANIFEST_INVALID")
    rows = witnesses.get("records")
    if not isinstance(rows, list) or len(rows) > 100:
        raise ValueError("SEC_WITNESS_COUNT_INVALID")
    by_acc: dict[str, list[dict]] = {}
    seen_ids: set[str] = set()
    for record in rows:
        if not isinstance(record, dict):
            raise TypeError("SEC_WITNESS_RECORD_INVALID")
        accession = record.get("accession")
        witness_id = record.get("witnessId")
        filename = record.get("sourceFilename")
        kind = record.get("sourceType")
        src_url = record.get("sourceUrl")
        if (not isinstance(accession, str) or accession not in allowed
                or not isinstance(witness_id, str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{4,64}", witness_id)
                or witness_id in seen_ids
                or kind not in SOURCE_TYPES
                or not isinstance(filename, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", filename)
                or not isinstance(src_url, str)):
            raise ValueError("SEC_WITNESS_RECORD_FIELDS_INVALID")
        parsed_url = urlsplit(src_url)
        if (parsed_url.scheme != "https" or not parsed_url.hostname
                or parsed_url.username or parsed_url.password
                or parsed_url.fragment or len(src_url) > 900
                or parsed_url.hostname.lower() in ("localhost", "127.0.0.1")
                or parsed_url.hostname.lower().endswith(".local")):
            raise ValueError("SEC_WITNESS_SOURCE_URL_INVALID")
        sha = record.get("sourceSha256")
        if not isinstance(sha, str) or not SHA.fullmatch(sha):
            raise ValueError("SEC_WITNESS_SHA_INVALID")
        file_path = source_dir / filename
        if (file_path.is_symlink() or not file_path.is_file()
                or file_path.stat().st_size > MAX_WITNESS_BYTES):
            raise ValueError("SEC_WITNESS_ORIGINAL_FILE_INVALID")
        if hashlib.sha256(file_path.read_bytes()).hexdigest() != sha:
            raise ValueError("SEC_WITNESS_ORIGINAL_SHA_MISMATCH")
        observed = _valid_timestamp(record.get("firstObservedAt"))
        published = record.get("publisherClaimedPublishedAt")
        published_date = _valid_timestamp(published) if published is not None else None
        if (published_date is not None and published_date > observed
                or observed.date().year < 2000):
            raise ValueError("SEC_WITNESS_CLOCK_ORDER_INVALID")
        # A modern HTML archive can display a historical date without proving
        # that it was continuously available at that date. No timestamp is
        # promoted without independent authenticated archive capture timing.
        seen_ids.add(witness_id)
        by_acc.setdefault(accession, []).append({
            "witnessId": witness_id,
            "sourceType": kind,
            "sourceUrl": src_url,
            "sourceSha256": sha,
            "firstObservedAtClaimed": observed.isoformat(),
            "publisherClaimedPublishedAt": (
                published_date.isoformat() if published_date else None
            ),
            "contemporaneousCaptureClockIndependentlyAuthenticated": False,
            "sourceContentIndependentlyAuthenticated": False,
            "usableAsPredecisionFeature": False,
            "reviewStatus": "SOURCE_BYTES_PINNED_HISTORICAL_CLOCK_NOT_AUTHENTICATED",
        })
    return by_acc


def reconcile_source_lineage(
    *,
    cohort: dict, cohort_sha: str,
    study: dict, study_sha: str,
    batch: dict, batch_sha: str, source_dir: Path,
    witnesses: dict | None = None, witness_sha: str | None = None,
) -> dict:
    tasks = _validate_cohort(cohort)
    if (not all(isinstance(s, str) and SHA.fullmatch(s)
                for s in (cohort_sha, study_sha, batch_sha))
            or cohort.get("multiyearStudySha256") != study_sha
            or not source_dir.is_dir() or source_dir.is_symlink()):
        raise ValueError("SEC_LINEAGE_INPUT_PROVENANCE_INVALID")
    if (study.get("schemaVersion") != "historical_sec_multiyear_price_study_v1"
            or study.get("status") !=
            "RAW_SEC_SOURCES_PINNED_RETROSPECTIVE_PRICE_OUTCOMES"
            or study.get("ticker") != "NVDA"
            or study.get("trainingReady") is not False
            or not isinstance(study.get("events"), list)
            or len(study["events"]) != study.get("eventCount")):
        raise ValueError("SEC_LINEAGE_STUDY_INVALID")
    prior: dict[str, str] = {}
    for event in study["events"]:
        acc, sha = event.get("accession"), event.get("secIndexSha256")
        if (not isinstance(acc, str) or not ACC.fullmatch(acc)
                or acc in prior or not isinstance(sha, str)
                or not SHA.fullmatch(sha)
                or event.get("issuerFilingIndexContentVerified") is not True
                or event.get("trainingEligible") is not False):
            raise ValueError("SEC_STUDY_ORIGINAL_SHA_INVALID")
        prior[acc] = sha
    if (batch.get("schemaVersion") != RESULT_SCHEMA
            or batch.get("status") !=
            "SEC_ORIGINAL_INDEX_SHA_PINNED_PUBLICATION_TIME_UNPROVEN"
            or batch.get("cohortSha256") != cohort_sha
            or batch.get("ticker") != "NVDA"
            or batch.get("trainingReady") is not False
            or batch.get("backtestReady") is not False
            or batch.get("modelPromotionAllowed") is not False
            or batch.get("knownFirstPublicAvailabilityCases") != 0
            or batch.get("secAcceptanceIsNotFirstPublicRelease") is not True
            or batch.get("secPublicDisseminationLagUnboundedHistorically") is not True):
        raise ValueError("SEC_BATCH_LINEAGE_MISMATCH")
    original = batch.get("sourceRecords")
    count = batch.get("capturedIndexCount")
    if (not isinstance(original, list) or not isinstance(count, int)
            or not 1 <= count <= len(tasks) or len(original) != count
            or batch.get("selectedTaskCount") != count
            or (batch.get("previouslyCachedCount", -1)
                + batch.get("newOriginalIndexCount", -1)) != count):
        raise ValueError("SEC_BATCH_CARDINALITY_INVALID")
    by_record = {item["accession"]: item for item in cohort["filingAvailabilityLedger"]}
    cases = []
    intersection = set()
    cached_count = 0
    for item, task in zip(original, tasks, strict=False):
        accession = task["accession"]
        if (not isinstance(item, dict)
                or item.get("accession") != accession
                or item.get("form") != task["form"]
                or item.get("filingDate") != task["filingDate"]
                or item.get("secIndexUrl") != task["secIndexUrl"]
                or item.get("historicalFirstPublicAvailabilityAt") is not None
                or item.get("firstIndependentHistoricalObservedAt") is not None
                or item.get("hasIndependentHistoricalPublicationClockEvidence") is not False
                or item.get("verifiedPreopenMarketAvailability") is not False
                or item.get("eligibleForNewsModelTraining") is not False
                or item.get("secReceiptClockOnly") is not True
                or item.get("secIndexMetadataCrossCheckedAgainstCohort") is not True):
            raise ValueError("SEC_CAPTURE_RECORD_PROVENANCE_INVALID")
        source_sha = item.get("secIndexSha256")
        if not isinstance(source_sha, str) or not SHA.fullmatch(source_sha):
            raise ValueError("SEC_CAPTURE_SOURCE_SHA_INVALID")
        cached = item.get("fetchedInThisRun") is False
        if type(item.get("fetchedInThisRun")) is not bool:
            raise ValueError("SEC_CAPTURE_FETCH_STATUS_INVALID")
        cached_count += cached
        claimed = by_record[accession]["secIndexSha256"]
        if claimed is not None and claimed != source_sha:
            raise ValueError("SEC_COHORT_AND_BATCH_SHA_CONFLICT")
        if accession in prior:
            intersection.add(accession)
            if prior[accession] != source_sha:
                raise ValueError("SEC_EXISTING_STUDY_BATCH_SHA_CONFLICT")
        local = source_dir / f"sec-{accession}-index.html"
        if (local.is_symlink() or not local.is_file()
                or local.stat().st_size > 1_500_000):
            raise ValueError("SEC_SOURCE_LOCAL_INVALID")
        source_raw = local.read_bytes()
        if hashlib.sha256(source_raw).hexdigest() != source_sha:
            raise ValueError("SEC_SOURCE_RAW_SHA_MISMATCH")
        accepted = _verify_index(source_raw, task)
        if accepted != item.get("secAcceptedLocalUnverifiedTimezone"):
            raise ValueError("SEC_SOURCE_ACCEPTED_CLOCK_MISMATCH")
        cases.append({
            "accession": accession,
            "form": task["form"],
            "filingDate": task["filingDate"],
            "secIndexSha256": source_sha,
            "alsoInOriginalTenCaseStudy": accession in prior,
            "reusedInCaptureBatch": cached,
            "receiptClockFromSecIndexTimezoneUnverified": accepted,
            "historicalPubliclyAvailableAt": None,
            "independentHistoricalFirstObservedAt": None,
            "predecisionEligibilityEstablished": False,
        })
    if cached_count != batch["previouslyCachedCount"]:
        raise ValueError("SEC_BATCH_REUSE_COUNT_MISMATCH")
    # Unauthenticated third-party source timestamps are preserved as claims,
    # never upgraded to independently proven past-first-public timestamps.
    if witnesses is not None and (
        not isinstance(witness_sha, str) or not SHA.fullmatch(witness_sha)
    ):
        raise ValueError("SEC_WITNESS_MANIFEST_SHA_INVALID")
    by_acc = _witness_records(
        witnesses, source_dir=source_dir,
        allowed={entry["accession"]: entry for entry in cases},
    )
    for case in cases:
        case["historicalWitnessClaims"] = by_acc.get(case["accession"], [])
        case["sourceWitnessClaimsCount"] = len(case["historicalWitnessClaims"])
    return {
        "schemaVersion": "sec_index_overlap_and_publication_witness_audit_v1",
        "status": "SEC_ORIGINAL_SHA_AUDITED_PUBLICATION_CLOCK_NOT_YET_VERIFIED",
        "ticker": "NVDA",
        "cohortSha256": cohort_sha,
        "multiyearStudySha256": study_sha,
        "originalIndexCaptureSha256": batch_sha,
        "independentWitnessManifestSha256": witness_sha,
        "originalTenCaseCount": len(prior),
        "capturedBatchCount": len(cases),
        "capturedBatchOverlapWithOriginalTen": len(intersection),
        "capturedBatchDistinctFromOriginalTen": len(cases) - len(intersection),
        "uniqueSecIndexCasesAcrossBoth": len(prior) + len(cases) - len(intersection),
        "overlappingAccessions": sorted(intersection),
        "witnessClaimCount": sum(len(v) for v in by_acc.values()),
        "independentlyVerifiedFirstPublicTimeCount": 0,
        "cases": cases,
        "allOriginalBytesRehashed": True,
        "priceOutcomesReadForSelection": False,
        "secReceiptIsNotPublicAvailableAt": True,
        "historicalWitnessClaimIsNotIndependentTimestampAttestation": True,
        "trainingReady": False,
        "backtestReady": False,
        "autoPromotionAllowed": False,
        "productionDbModified": False,
        "finalTestTouched": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline SHA cross-check + historical publication witness triage"
    )
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--witnesses", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    try:
        if not args.preview and args.output is None:
            raise ValueError("SEC_LINEAGE_OUTPUT_REQUIRED")
        cohort, cohort_sha = _read_cohort(args.cohort)
        study, study_sha = _read_study(args.study)
        batch, batch_sha = _read_json(args.batch, MAX_REPORT_BYTES)
        witness, witness_sha = (
            _read_json(args.witnesses, MAX_REPORT_BYTES)
            if args.witnesses is not None else (None, None)
        )
        report = reconcile_source_lineage(
            cohort=cohort, cohort_sha=cohort_sha,
            study=study, study_sha=study_sha, batch=batch,
            batch_sha=batch_sha, source_dir=args.source_dir,
            witnesses=witness, witness_sha=witness_sha,
        )
        if not args.preview:
            _write(args.output, report)
        print(json.dumps({
            "status": report["status"],
            "studyOriginalCases": report["originalTenCaseCount"],
            "batchOriginalIndexes": report["capturedBatchCount"],
            "batchOverlap": report["capturedBatchOverlapWithOriginalTen"],
            "batchNewDistinctCases": report["capturedBatchDistinctFromOriginalTen"],
            "totalUniqueIndexes": report["uniqueSecIndexCasesAcrossBoth"],
            "historicalFirstPublicTimesVerified": 0,
            "trainingReady": False,
            "outputCreated": not args.preview,
        }))
    except (OSError, ValueError, TypeError, KeyError, UnicodeError,
            json.JSONDecodeError):
        parser.exit(2, "sec-source-lineage: ORIGINAL_OR_PUBLICATION_EVIDENCE_INVALID\n")


if __name__ == "__main__":
    main()
