"""Optional substantive original-source text for direct sector analysis."""
from __future__ import annotations

import hashlib
from typing import Any

from tris_market_event_intelligence.event_evidence_policy import plain_text


def use_bok_attachments(articles: list[dict], report: dict[str, Any], frozen_hash: str) -> list[dict]:
    if (report.get("schemaVersion") != "bok_attachment_evidence_v1"
            or report.get("originalHoldoutSha256") != frozen_hash):
        raise ValueError("BOK_ATTACHMENT_FROZEN_COHORT_MISMATCH")
    rows = report.get("articles")
    if not isinstance(rows, list) or len(rows) != report.get("targetedCount"):
        raise ValueError("BOK_ATTACHMENT_COHORT_SIZE_MISMATCH")
    by_id = {row["articleId"]: row for row in rows}
    available = {row["articleId"] for row in articles}
    if len(by_id) != len(rows) or not set(by_id).issubset(available):
        raise ValueError("BOK_ATTACHMENT_ARTICLE_ID_MISMATCH")
    output: list[dict] = []
    for article in articles:
        evidence = by_id.get(article["articleId"])
        if evidence is None:
            output.append(article)
            continue
        if not str(article.get("source") or "").startswith("bok_"):
            raise ValueError("BOK_ATTACHMENT_SOURCE_MISMATCH")
        original = plain_text(article.get("summary") or "")
        fragments: list[str] = []
        for row in evidence.get("attachments", []):
            if row.get("extractionStatus") != "TEXT_EXTRACTED":
                continue
            segment = row.get("extractedText")
            if not isinstance(segment, str) or hashlib.sha256(segment.encode()).hexdigest() != row.get("extractedTextSha256"):
                raise ValueError("BOK_ATTACHMENT_CONTENT_HASH_MISMATCH")
            fragments.append(segment[:11000])
            if len(fragments) == 2:
                break
        if fragments:
            enriched = original[:1800] + " OFFICIAL BOK ATTACHMENT TEXT: " + " ".join(fragments)
            output.append({**article, "summary": enriched, "attachedEvidenceSource": "BOK_OFFICIAL_ATTACHMENT"})
        else:
            output.append(article)
    return output
