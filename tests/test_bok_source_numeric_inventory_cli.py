"""Offline numeric inventory: never confuse source presence with correct interpretation."""
import hashlib

import pytest

from tris_market_event_intelligence.bok_source_numeric_inventory_cli import (
    audit_attachment_report,
    extract_quantities,
    scale_value,
)


def test_won_scales_and_no_guessing_for_compound_amounts():
    assert scale_value("2.5", "조원") == "2500000000000.0"
    assert scale_value("2939", "만원") == "29390000"
    assert scale_value("30", "%") is None
    assert scale_value("3", "억2939만원") is None


def test_verbatim_spans_comparison_and_uncertainty():
    source = "전기 대비 물가상승률이 2.5% 증가하였고, 전년 대비 매출은 3억원 감소했다."
    rows = extract_quantities(source)
    assert rows
    assert all(source[x["sourceStart"]:x["sourceEnd"]] == x["verbatim"] for x in rows)
    assert all(x["economicMeaningVerified"] is False for x in rows)
    assert any(x["normalizedKrw"] == "300000000" for x in rows)
    assert any("QOQ" in x["comparisonCandidates"] for x in rows)
    assert any("YOY" in x["comparisonCandidates"] for x in rows)


def test_saved_attachment_text_hash_must_match():
    source = "2026년 분기 실적은 3억원이며 전기 대비 2.5% 증가했다."
    entry = {
        "extractionStatus": "TEXT_EXTRACTED", "sourceUrl": "https://www.bok.or.kr/a",
        "extractedText": source,
        "extractedTextSha256": hashlib.sha256(source.encode()).hexdigest(),
    }
    report = {
        "schemaVersion": "bok_attachment_evidence_v1", "targetedCount": 1,
        "originalHoldoutSha256": "frozen",
        "articles": [{"articleId": 210, "attachments": [entry]}],
    }
    inventory = audit_attachment_report(report)
    assert inventory["statusCounts"] == {"NUMERIC_EVIDENCE_INVENTORIED": 1}
    assert inventory["pointInTimeCertified"] is False
    assert inventory["originalPredictionsModified"] is False
    entry["extractedText"] = "modified"
    with pytest.raises(ValueError, match="EXTRACTED_TEXT_HASH_MISMATCH"):
        audit_attachment_report(report)


def test_unreadable_attachment_is_not_factual_evidence():
    report = {
        "schemaVersion": "bok_attachment_evidence_v1", "targetedCount": 1,
        "articles": [{"articleId": 210, "attachments": [
            {"extractionStatus": "HWP_OLE_EXTRACTOR_UNAVAILABLE"}
        ]}],
    }
    result = audit_attachment_report(report)
    assert result["statusCounts"] == {"NO_READABLE_ATTACHMENT": 1}
    assert result["numericCandidates"] == 0


def test_duplicate_article_ids_rejected():
    report = {
        "schemaVersion": "bok_attachment_evidence_v1", "targetedCount": 2,
        "articles": [{"articleId": 1}, {"articleId": 1}],
    }
    with pytest.raises(ValueError, match="DUPLICATE_ARTICLE_ID"):
        audit_attachment_report(report)
