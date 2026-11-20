"""Substantive source text must be joined to its original article only."""
import hashlib

import pytest

from tris_market_event_intelligence.direct_sector_source_enrichment import (
    use_bok_attachments,
)
from tris_market_event_intelligence.direct_sector_impact_cli import analyze


def example():
    body = "2026년 3월 통화 및 유동성 자료는 한국은행 공식 발표이다."
    original = [{"articleId": 210, "source": "bok_release", "title": "M2", "summary": body}]
    text = "반도체 설비투자 증가율은 15% 상승했고 은행 대출 증가율은 2% 하락했다."
    report = {
        "schemaVersion": "bok_attachment_evidence_v1",
        "originalHoldoutSha256": "frozen", "targetedCount": 1,
        "articles": [{"articleId": 210, "attachments": [{
            "extractionStatus": "TEXT_EXTRACTED",
            "extractedText": text,
            "extractedTextSha256": hashlib.sha256(text.encode()).hexdigest(),
        }]}],
    }
    return original, report


def test_substantive_bok_evidence_is_in_model_input():
    originals, report = example()
    enriched = use_bok_attachments(originals, report, "frozen")
    assert originals[0]["summary"] != enriched[0]["summary"]
    assert "반도체 설비투자 증가율은 15% 상승" in enriched[0]["summary"]
    assert enriched[0]["attachedEvidenceSource"] == "BOK_OFFICIAL_ATTACHMENT"


def test_wrong_cohort_or_changed_content_is_not_accepted():
    originals, report = example()
    with pytest.raises(ValueError, match="BOK_ATTACHMENT_FROZEN_COHORT_MISMATCH"):
        use_bok_attachments(originals, report, "other")
    report["articles"][0]["attachments"][0]["extractedText"] = "tampered"
    with pytest.raises(ValueError, match="BOK_ATTACHMENT_CONTENT_HASH_MISMATCH"):
        use_bok_attachments(originals, report, "frozen")


def test_reject_unknown_attachment_article_identity():
    originals, report = example()
    report["articles"][0]["articleId"] = 1000
    with pytest.raises(ValueError, match="BOK_ATTACHMENT_ARTICLE_ID_MISMATCH"):
        use_bok_attachments(originals, report, "frozen")


def test_no_model_prediction_made_during_source_join():
    originals, report = example()
    enriched = use_bok_attachments(originals, report, "frozen")
    assert "price" not in enriched[0]
    assert "prediction" not in enriched[0]
