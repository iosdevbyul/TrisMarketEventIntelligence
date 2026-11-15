"""Link verified issuer document facts to an unverified GDELT story inventory.

A source's calendar date and an archived GKG URL are NOT proof of its
earliest publicly tradable timestamp. The resulting packet must remain a
retrospective study; no historical news is backdated for model input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

MAX_BYTES = 700_000
REVIEW_SCHEMA = "historical_story_review_v1"
SOURCE_SCHEMA = "historical_issuer_primary_evidence_v1"
RESULT_SCHEMA = "historical_source_corrob_review_v1"


def _read(path: Path) -> tuple[dict, str]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError("Bounded non-symlink input required")
    raw = path.read_bytes()
    if len(raw) > MAX_BYTES:
        raise ValueError("Source input size changed")
    doc = json.loads(raw)
    if not isinstance(doc, dict):
        raise TypeError("Expected JSON object")
    return doc, hashlib.sha256(raw).hexdigest()


def corroborate_sources(*, review: dict, review_sha: str,
                        issuer: dict, issuer_sha: str) -> dict:
    source = issuer.get("primarySource")
    if (issuer.get("schemaVersion") != SOURCE_SCHEMA
            or review.get("schemaVersion") != REVIEW_SCHEMA
            or review.get("status") != "URL_METADATA_ONLY_REVIEW_REQUIRED"
            or not isinstance(source, dict)
            or issuer.get("ticker") != "NVDA"
            or issuer.get("shockDate") != "2018-11-16"
            or review.get("ticker") != issuer["ticker"]
            or review.get("shockDate") != issuer["shockDate"]
            or review.get("retrospectiveOutcomeKnown") is not True
            or review.get("notIndependentGold") is not True
            or review.get("articleTextsFetched") != 0
            or review.get("priceCausalityVerified") is not False
            or review.get("promotionAllowed") is not False):
        raise ValueError("Issuer and historical GDELT study contract mismatch")
    primary_url = source.get("sourceUrl")
    if (not isinstance(primary_url, str) or urlsplit(primary_url).scheme != "https"
            or urlsplit(primary_url).hostname != "nvidianews.nvidia.com"
            or source.get("sourceType") != "ISSUER_ORIGINAL_PRESS_RELEASE"
            or source.get("dateDisplayedOnOriginal") is not True
            or source.get("publisherFirstPublicationTimeUtc") is not None
            or source.get("firstTradableAvailabilityVerified") is not False
            or source.get("originalBytesPinned") is not False
            or issuer.get("truthControls", {}).get(
                "sufficientForProspectiveTraining"
            ) is not False):
        raise ValueError("Issuer has an ungrounded publication-time claim")
    release_day = date.fromisoformat(source["announcementDate"])
    shock_day = date.fromisoformat(issuer["shockDate"])
    if release_day >= shock_day:
        raise ValueError("Expected a previous-date original issuer disclosure")
    facts = issuer.get("facts")
    if (not isinstance(facts, list) or len(facts) != 3
            or len({item.get("id") for item in facts if isinstance(item, dict)}) != 3
            or any(not isinstance(item, dict)
                   or not isinstance(item.get("statement"), str)
                   or len(item["statement"]) < 30
                   or item.get("nature") not in {
                       "REPORTED_RESULT", "FORWARD_LOOKING_GUIDANCE",
                       "ISSUER_EXPLANATION",
                   } for item in facts)):
        raise ValueError("Require unique original issuer facts")
    references = review.get("references")
    if (not isinstance(references, list)
            or len(references) != review.get("referenceCount")
            or not 1 <= len(references) <= 400):
        raise ValueError("Invalid historical GKG candidate inventory")
    candidate_ids = set()
    urls: set[str] = set()
    metadata_only = []
    for item in references:
        if not isinstance(item, dict):
            raise TypeError("Invalid GKG candidate")
        ref_id, url = item.get("referenceId"), item.get("sourceUrl")
        if (not isinstance(ref_id, str) or not isinstance(url, str)
                or ref_id in candidate_ids or url in urls
                or item.get("articleTextAvailable") is not False
                or item.get("publisherFirstPublishedAt") is not None
                or item.get("earliestTradableAvailabilityVerified") is not False
                or item.get("issuerMatchIndependentlyVerified") is not False
                or item.get("candidateCategory") not in {
                    "COMPANY_EARNINGS_OR_GUIDANCE", "COMPANY_OTHER",
                    "COMPETITOR_OR_SECTOR", "MACRO_OR_MARKET",
                    "OTHER_OR_UNCLEAR",
                }):
            raise ValueError("GKG metadata is not verified issuer evidence")
        candidate_ids.add(ref_id)
        urls.add(url)
        metadata_only.append({
            "referenceId": ref_id,
            "sourceUrl": url,
            "candidateCategory": item["candidateCategory"],
            "publisherClockVerified": False,
            "issuerLinkVerified": False,
            "articleTextRead": False,
            "usableAtHistoricalPredictionCutoff": False,
        })
    groups = review.get("candidateStoryGroups")
    if not isinstance(groups, list):
        raise ValueError("Missing original GKG duplicate-group evidence")
    for group in groups:
        if (not isinstance(group, dict)
                or not isinstance(group.get("referenceIds"), list)
                or not set(group["referenceIds"]) <= candidate_ids
                or group.get("sameUnderlyingStoryVerified") is not False):
            raise ValueError("Cannot promote heuristic duplicate groups")
    categories = dict(Counter(item["candidateCategory"] for item in metadata_only))
    exact_matches = [
        item["referenceId"] for item in metadata_only
        if item["sourceUrl"] == primary_url
    ]
    return {
        "schemaVersion": RESULT_SCHEMA,
        "status": "OFFICIAL_DATE_AND_FACTS_CORROBORATED_CLOCK_UNVERIFIED",
        "eventId": issuer["eventId"],
        "ticker": issuer["ticker"],
        "shockDate": issuer["shockDate"],
        "historicalReviewSha256": review_sha,
        "issuerEvidenceSha256": issuer_sha,
        "issuerSource": {
            "sourceId": source["id"],
            "sourceUrl": primary_url,
            "documentDate": source["announcementDate"],
            "verifiedDirectIssuerAssociation": True,
            "publisherFirstPublicationClockVerified": False,
            "preOpenAvailabilityVerified": False,
            "sourceOriginalBytesPinned": False,
            "facts": facts,
        },
        "gkgCandidateCount": len(metadata_only),
        "gkgCandidateCategories": dict(sorted(categories.items())),
        "gkgOfficialExactUrlMatchRefs": exact_matches,
        "gkgCandidates": metadata_only,
        "gkgCandidateGroupCount": len(groups),
        "gkgDateIsArchiveBatchNotArticlePublication": True,
        "independentPublisherTimeChecked": False,
        "issuerReleaseEarlierCalendarDayThanShock": True,
        "sourceDocumentDayIsNotClockEvidence": True,
        "reconstructedPastNewsAvailability": False,
        "priceImpactCausalityVerified": False,
        "replayTrainingEligible": False,
        "prospectiveForecastProduced": False,
        "marketReturnOutcomeObservedHere": False,
        "modelCalls": 0,
        "modelPromotionAllowed": False,
        "productionDbModified": False,
    }


def _write(path: Path, result: dict) -> None:
    if path.is_symlink() or path.exists() or not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError("Expected new private output")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify original issuer facts and GDELT candidates")
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--issuer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        review, review_sha = _read(args.review)
        issuer, issuer_sha = _read(args.issuer)
        report = corroborate_sources(
            review=review, review_sha=review_sha,
            issuer=issuer, issuer_sha=issuer_sha,
        )
        _write(args.output, report)
        print(json.dumps({
            "status": report["status"],
            "shockDate": report["shockDate"],
            "gkgCandidates": report["gkgCandidateCount"],
            "categoryCounts": report["gkgCandidateCategories"],
            "officialSourceFacts": len(report["issuerSource"]["facts"]),
            "exactOfficialUrlMatches": len(report["gkgOfficialExactUrlMatchRefs"]),
            "publisherClockVerified": False,
            "trainingReady": False,
            "modelCalls": 0,
            "outputCreated": True,
        }))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        parser.exit(2, "historical-issuer-corroborate: INVALID_SOURCE_OR_PROVENANCE\n")


if __name__ == "__main__":
    main()
