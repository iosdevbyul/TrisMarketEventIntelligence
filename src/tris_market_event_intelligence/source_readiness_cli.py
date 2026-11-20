"""Read-only gate: source completeness is not certified first-publication timing."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from tris_market_event_intelligence.event_evidence_policy import content_quality, plain_text
from tris_market_event_intelligence.source_grounded_semantic_analysis import issuer_provenance

SHA = re.compile(r"[0-9a-f]{64}\Z")
ATTACHMENTS = ("붙임", "첨부파일", "첨부자료", "별첨", "attachment", "attached", "exhibit 99", "exhibit 10")
ATTACHMENT_ONLY = re.compile(
    r"(?:자세한 내용|상세 내용|구체적인 내용|세부 내용|통계표|원문|전문)"
    r".{0,35}(?:붙임|첨부파일|첨부자료|별첨|attachment|attached)"
    r"|(?:붙임|첨부파일|첨부자료|별첨).{0,35}(?:참고|참조|확인)",
    re.IGNORECASE,
)


def attachment_evidence(body: str, quality: dict[str, Any]) -> dict[str, Any]:
    """Distinguish a mention from a missing-data pointer; never certify attachments."""
    mentions = [marker for marker in ATTACHMENTS if marker in body.casefold()]
    pointers = list(ATTACHMENT_ONLY.finditer(body))
    pointer = pointers[0] if pointers else None
    excerpt = (
        body[max(0, pointer.start() - 45): min(len(body), pointer.end() + 45)]
        if pointer else ""
    )
    if quality.get("reasonCode") == "ATTACHMENT_CONTENT_MISSING":
        kind = "SOURCE_BODY_POINTS_TO_MISSING_ATTACHMENT"
    elif pointer:
        kind = "ATTACHMENT_POINTER_NEEDS_EVIDENCE_REVIEW"
    elif mentions:
        kind = "ATTACHMENT_MENTION_NEEDS_EVIDENCE_REVIEW"
    else:
        kind = "NO_ATTACHMENT_REFERENCE_DETECTED"
    return {
        "status": kind,
        "markers": mentions,
        "supportingText": excerpt,
        "attachmentBytesIndependentlyVerified": False,
    }

NUMBERS = re.compile(r"(?<!\w)(?:[$₩€£]\s*)?\d[\d,.]*(?:\s*(?:%|퍼센트|bp|bps|조|억|만|원|달러|USD|KRW|billion|million))?", re.IGNORECASE)


def timestamp_status(value: Any) -> str:
    try:
        parsed = datetime.fromisoformat(value)
        return "REPORTED_TIMESTAMP_ONLY" if parsed.tzinfo else "TIMESTAMP_TIMEZONE_MISSING"
    except (ValueError, AttributeError):
        return "INVALID_REPORTED_TIMESTAMP"


def sec_exhibits(article: dict, row: dict) -> tuple[list[str], dict]:
    source = str(article.get("source", ""))
    if not source.startswith("sec_8k_"):
        return [], {"status": "NOT_SEC_FILING"}
    audit = row.get("officialSECExhibitEvidence") or {}
    prefix = str(article.get("sourceUrl") or "").rsplit("/", 1)[0] + "/"
    verified = []
    for exhibit in audit.get("acquiredExhibits", []):
        if (isinstance(exhibit, dict)
                and str(exhibit.get("sourceUrl", "")).startswith(prefix)
                and SHA.fullmatch(str(exhibit.get("rawSha256", "")))
                and SHA.fullmatch(str(exhibit.get("extractedTextSha256", "")))
                and isinstance(exhibit.get("extractedChars"), int)
                and exhibit["extractedChars"] >= 100):
            verified.append(exhibit.get("exhibitType"))
    title = str(article.get("title", "")).casefold()
    requirement = ("99.1",) if "item 2.02" in title else (
        ("10.1", "99.1") if "item 1.01" in title else ()
    )
    missing = bool(requirement and not set(requirement).intersection(verified))
    return (
        ["REQUIRED_SEC_EXHIBIT_NOT_VERIFIED"] if missing else [],
        {"status": "REQUIRED_EXHIBIT_MISSING" if missing else "EVIDENCE_CHECKED",
         "requiredAnyOf": list(requirement), "verifiedExhibitTypes": verified,
         "retrievalStatus": audit.get("status", "NOT_RECORDED")}
    )


def evaluate_source(article: dict, row: dict) -> dict:
    identity = ("articleId", "title", "source", "sourceUrl", "publishedAtReported")
    if any(article.get(key) != row.get(key) for key in identity):
        raise ValueError("SEMANTIC_SOURCE_IDENTITY_MISMATCH")
    body = plain_text(article.get("summary"))
    digest = hashlib.sha256(body.encode()).hexdigest()
    expected = row.get("originalSourceBodySha256") or row.get("sourceBodySha256")
    if digest != expected:
        raise ValueError("SEMANTIC_ORIGINAL_SOURCE_HASH_MISMATCH")
    quality = content_quality(article)
    errors = []
    if quality["status"] != "ELIGIBLE":
        errors.append(quality["reasonCode"])
    url = urlparse(str(article.get("sourceUrl") or ""))
    if url.scheme != "https" or not url.netloc or url.username or url.password:
        errors.append("SOURCE_URL_INVALID")
    sec = str(article.get("source", "")).startswith("sec_8k_")
    if sec and issuer_provenance(article)["status"] != "OFFICIAL_SEC_ISSUER_CIK_BINDING":
        errors.append("SEC_ISSUER_ARCHIVE_NOT_VERIFIED")
    ts = timestamp_status(article.get("publishedAtReported"))
    if ts != "REPORTED_TIMESTAMP_ONLY":
        errors.append(ts)
    exhibit_errors, exhibits = sec_exhibits(article, row)
    errors.extend(exhibit_errors)
    attachment = attachment_evidence(body, quality)
    references_attachment = bool(attachment["markers"])
    if not sec and references_attachment:
        if attachment["status"] == "SOURCE_BODY_POINTS_TO_MISSING_ATTACHMENT":
            errors.append("NON_SEC_ATTACHMENT_CONTENT_UNVERIFIED")
        else:
            errors.append("NON_SEC_ATTACHMENT_REVIEW_REQUIRED")
    ready = not errors
    needs_review = errors == ["NON_SEC_ATTACHMENT_REVIEW_REQUIRED"]
    return {
        "articleId": article["articleId"], "title": article["title"],
        "source": article["source"], "sourceUrl": article["sourceUrl"],
        "publishedAtReported": article["publishedAtReported"],
        "originalSourceBodySha256": digest, "contentQuality": quality,
        "sourceReadiness": (
            "SOURCE_READY" if ready else
            "SOURCE_REVIEW_REQUIRED" if needs_review else "SOURCE_INCOMPLETE"
        ),
        "blockingReasons": sorted(set(errors)), "secExhibitEvidence": exhibits,
        "attachmentReferenced": references_attachment,
        "attachmentEvidence": attachment,
        "numericSourceTokens": len(NUMBERS.findall(body)),
        "numericValuesAndUnitsVerified": False,
        "timestampEvidenceStatus": ts,
        "firstPublicReleaseStatus": "PUBLICATION_TIME_UNVERIFIED",
        "sourceWasPointInTimeCertified": False,
        "eligibleForFactualInterpretation": ready,
        "eligibleForCertifiedHistoricalForecast": False,
    }


def audit_sources(articles: list[dict], semantic: dict) -> dict:
    rows = semantic.get("articles")
    if not isinstance(rows, list) or len(rows) != len(articles):
        raise ValueError("FROZEN_COHORT_LENGTH_MISMATCH")
    by_id = {row["articleId"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != {a["articleId"] for a in articles}:
        raise ValueError("FROZEN_COHORT_ARTICLE_SET_MISMATCH")
    result = [evaluate_source(a, by_id[a["articleId"]]) for a in articles]
    return {
        "schemaVersion": "historical_source_readiness_v1",
        "originalHoldoutSha256": semantic.get("originalHoldoutSha256"),
        "articleCount": len(result),
        "sourceReadinessCounts": dict(Counter(a["sourceReadiness"] for a in result)),
        "blockingReasonCounts": dict(Counter(e for a in result for e in a["blockingReasons"])),
        "firstPublicationPITCertifiedCount": 0,
        "priceDataAccessed": False, "modelWasRerun": False,
        "originalPredictionsModified": False,
        "articles": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-result", type=Path, required=True)
    parser.add_argument("--semantic-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for p in (args.holdout_result, args.semantic_report):
        if not p.is_file() or p.is_symlink():
            parser.error(f"Missing or unsafe input: {p}")
    if args.output.exists() or args.output.is_symlink():
        parser.error("Refusing to overwrite report")
    from tris_market_event_intelligence.source_grounded_semantic_replay_cli import source_items
    articles, holdout_sha, _ = source_items(args.holdout_result)
    semantic = json.loads(args.semantic_report.read_text(encoding="utf-8"))
    if semantic.get("schemaVersion") != "frozen_thirty_source_grounded_semantic_v1":
        parser.error("Unsupported semantic schema")
    if semantic.get("originalHoldoutSha256") != holdout_sha:
        parser.error("Source and semantic holdout hashes differ")
    result = audit_sources(articles, semantic)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as writer:
        json.dump(result, writer, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in result.items() if k != "articles"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
