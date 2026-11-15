"""Bounded, resumable retrieval of historical NVDA GDELT GKG *references*.

GDELT 15-minute archive slots are evidence of the archive batch, never
publisher-first-publication clocks or verified contemporaneous training input.
This is an explicitly invoked historical research workflow, not 24/7 polling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import date
from pathlib import Path

import httpx

from tris_market_event_intelligence.historical_gdelt_scout import (
    _read_study,
    _slots,
    scout_historical_gdelt,
)

MAX_EXISTING = 1_000_000
MAX_BATCH = 3


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_receipt(path: Path, *, study_sha: str, rank: int, day: str) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_EXISTING:
        raise ValueError("Invalid prior private archive receipt")
    packet = json.loads(path.read_bytes())
    if not isinstance(packet, dict) or any((
        packet.get("schemaVersion") != "gdelt_gkg_historical_scout_v1",
        packet.get("status") != "UNVERIFIED_GDELT_DOCUMENT_REFERENCES",
        packet.get("studySha256") != study_sha,
        packet.get("ticker") != "NVDA",
        packet.get("benchmark") != "SPY",
        packet.get("shockRank") != rank,
        packet.get("shockDate") != day,
        packet.get("retrospectiveOutcomeKnown") is not True,
        packet.get("predictionOrTrainingPerformed") is not False,
        packet.get("earliestArticleAvailabilityVerified") is not False,
        packet.get("sourceArticleRightsVerified") is not False,
        packet.get("promotionAllowed") is not False,
        packet.get("archivesFetched") != 4,
    )):
        raise ValueError("Existing archive receipt does not match pinned study")
    archives = packet.get("archives")
    if not isinstance(archives, list) or len(archives) != 4:
        raise ValueError("Incomplete prior archive receipt")
    plan = packet.get("archivePlan")
    if not isinstance(plan, list) or len(plan) != 4:
        raise ValueError("Unpinned prior archive plan")
    totals = 0
    for archived, planned, (expected_slot, expected_url) in zip(
        archives, plan, _slots(date.fromisoformat(day)), strict=True,
    ):
        if (not isinstance(archived, dict) or not isinstance(planned, dict)
                or archived.get("gkgSlot") != planned.get("slotUtc")
                or planned.get("slotUtc") != expected_slot.isoformat()
                or planned.get("url") != expected_url
                or archived.get("archiveSlotNotPublicationTimestamp") is not True
                or archived.get("earliestArticleAvailabilityVerified") is not False):
            raise ValueError("Invalid prior GDELT archive time provenance")
        digest = archived.get("archiveSha256")
        refs = archived.get("matchingDocumentReferences")
        count = archived.get("matchingDocumentsTotal")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
                or type(count) is not int or count < 0
                or not isinstance(refs, list) or len(refs) > 50
                or len(refs) > count):
            raise ValueError("Invalid GDELT archive reference receipt")
        for ref in refs:
            if (not isinstance(ref, dict)
                    or ref.get("gkgBatchSlotUtc") != archived["gkgSlot"]
                    or ref.get("articleTextRetrieved") is not False
                    or ref.get("issuerVerified") is not False
                    or ref.get("independentCausalAttribution") is not False
                    or not isinstance(ref.get("sourceUrl"), str)):
                raise ValueError("GDELT references cannot imply verified news causality")
        totals += count
    return {"matchedReferences": totals, "receiptSha256": _sha(path.read_bytes())}


def _write(path: Path, report: dict) -> None:
    if path.exists() or path.is_symlink() or path.parent.is_symlink():
        raise ValueError("Existing private file cannot be overwritten")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def replay_archive_coverage(
    *, study: Path, output_dir: Path, start_rank: int, count: int,
    run: bool = False, client: httpx.Client | None = None,
) -> dict:
    if (type(start_rank) is not int or not 1 <= start_rank <= 100
            or type(count) is not int or not 1 <= count <= MAX_BATCH
            or start_rank + count > 101):
        raise ValueError("Only bounded top-100 rank windows supported")
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError("Existing non-symlink local data directory required")
    packet, source_sha = _read_study(study)
    plans = []
    for rank in range(start_rank, start_rank + count):
        day = packet["topEvents"][rank - 1]["date"]
        if not isinstance(day, str):
            raise ValueError("Invalid pinned research date")
        path = output_dir / f"nvda-gdelt-gkg-rank{rank}.json"
        if path.is_symlink():
            raise ValueError("Symlink receipts prohibited")
        if path.exists():
            checked = _read_receipt(
                path, study_sha=source_sha, rank=rank, day=day,
            )
            plans.append({
                "shockRank": rank, "shockDate": day, "action": "REUSE_EXISTING",
                **checked,
            })
        else:
            # We validate each GDELT slot through the existing scout in run
            # mode; preview remains entirely offline.
            plans.append({
                "shockRank": rank, "shockDate": day,
                "action": "FETCH_FOUR_SLOTS" if run else "WOULD_FETCH_FOUR_SLOTS",
                "matchedReferences": None, "receiptSha256": None,
            })
    # A single shared read-only HTTP client remains bounded and closed here.
    owns_client = client is None
    session = client or httpx.Client(timeout=httpx.Timeout(45), trust_env=False)
    try:
        if run:
            for entry in plans:
                if entry["action"] == "REUSE_EXISTING":
                    continue
                rank = entry["shockRank"]
                report = scout_historical_gdelt(
                    study_path=study, rank=rank, run=True, client=session,
                )
                # Check all metadata before committing a new immutable receipt.
                path = output_dir / f"nvda-gdelt-gkg-rank{rank}.json"
                _write(path, report)
                entry.update(_read_receipt(
                    path, study_sha=source_sha, rank=rank,
                    day=entry["shockDate"],
                ))
                entry["action"] = "FETCHED_AND_PINNED"
    finally:
        if owns_client:
            session.close()
    return {
        "schemaVersion": "historical_gdelt_replay_coverage_v1",
        "status": "UNVERIFIED_ARCHIVE_REFERENCES_SCOUTED" if run else "PREVIEW_ONLY",
        "studySha256": source_sha,
        "shockRanksPlanned": len(plans),
        "archiveSlotsEach": 4,
        "newReceipts": sum(p["action"] == "FETCHED_AND_PINNED" for p in plans),
        "reusedReceipts": sum(p["action"] == "REUSE_EXISTING" for p in plans),
        "referenceCandidates": sum(p["matchedReferences"] or 0 for p in plans),
        "entries": plans,
        "publisherTimesVerified": False,
        "trainingReady": False,
        "priceCausalityEstablished": False,
        "predictionReady": False,
        "productionDbModified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resumable opt-in rank-bounded GDELT archive references"
    )
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--start-rank", type=int, default=1)
    parser.add_argument("--count", type=int, default=1)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    try:
        report = replay_archive_coverage(
            study=args.study, output_dir=args.data_dir,
            start_rank=args.start_rank, count=args.count, run=args.run,
        )
        print(json.dumps({
            "status": report["status"],
            "ranks": [p["shockRank"] for p in report["entries"]],
            "newReceipts": report["newReceipts"],
            "reusedReceipts": report["reusedReceipts"],
            "referenceCandidates": report["referenceCandidates"],
            "trainingReady": False, "predictionReady": False,
            "receipts": [
                {"rank": p["shockRank"], "date": p["shockDate"],
                 "status": p["action"]}
                for p in report["entries"]
            ],
        }))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        parser.exit(2, "historical-gdelt-replay: INVALID_SOURCE_OR_ARCHIVE\n")


if __name__ == "__main__":
    main()
