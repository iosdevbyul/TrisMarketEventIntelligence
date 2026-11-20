"""Source completeness must not masquerade as a historical publication proof."""
import hashlib

import pytest

from tris_market_event_intelligence.source_readiness_cli import audit_sources, evaluate_source


def inputs(source="bok_statistics", title="2026년 2분기 경제 통계 발표"):
    body = "2026년 2분기 경제 통계에 따르면 생산이 전기 대비 2.0% 증가했습니다. 이 통계는 공식 조사에 근거합니다."
    article = {
        "articleId": 1, "source": source, "title": title,
        "sourceUrl": "https://www.bok.or.kr/example",
        "summary": body, "publishedAtReported": "2026-11-01T10:00:00+09:00",
    }
    row = {key: article[key] for key in ("articleId", "source", "title", "sourceUrl", "publishedAtReported")}
    row["originalSourceBodySha256"] = hashlib.sha256(body.encode()).hexdigest()
    return article, row


def test_source_ready_never_means_pit_certified():
    article, row = inputs()
    result = evaluate_source(article, row)
    assert result["sourceReadiness"] == "SOURCE_READY"
    assert result["numericValuesAndUnitsVerified"] is False
    assert result["firstPublicReleaseStatus"] == "PUBLICATION_TIME_UNVERIFIED"
    assert result["eligibleForCertifiedHistoricalForecast"] is False


def test_bok_attachment_reference_blocks():
    article, row = inputs()
    article["summary"] += " 상세 통계는 붙임 자료 참조."
    row["originalSourceBodySha256"] = hashlib.sha256(article["summary"].encode()).hexdigest()
    assert "NON_SEC_ATTACHMENT_CONTENT_UNVERIFIED" in evaluate_source(article, row)["blockingReasons"]


def test_earnings_cover_requires_verified_exhibit():
    article, row = inputs("sec_8k_2488", "AMD Item 2.02")
    article["sourceUrl"] = "https://www.sec.gov/Archives/edgar/data/2488/000000248826000072/amd.htm"
    row["sourceUrl"] = article["sourceUrl"]
    assert "REQUIRED_SEC_EXHIBIT_NOT_VERIFIED" in evaluate_source(article, row)["blockingReasons"]
    row["officialSECExhibitEvidence"] = {
        "status": "EXHIBITS_ACQUIRED",
        "acquiredExhibits": [{
            "exhibitType": "99.1",
            "sourceUrl": "https://www.sec.gov/Archives/edgar/data/2488/000000248826000072/ex99.htm",
            "rawSha256": "a" * 64, "extractedTextSha256": "b" * 64,
            "extractedChars": 150,
        }],
    }
    assert evaluate_source(article, row)["sourceReadiness"] == "SOURCE_READY"
    row["officialSECExhibitEvidence"]["acquiredExhibits"][0]["sourceUrl"] = "https://example.org/fake.htm"
    assert evaluate_source(article, row)["sourceReadiness"] == "SOURCE_INCOMPLETE"


def test_mutations_fail_closed():
    article, row = inputs()
    article["summary"] += " changed"
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        evaluate_source(article, row)
    article, row = inputs()
    row["title"] = "mismatched"
    with pytest.raises(ValueError, match="IDENTITY_MISMATCH"):
        evaluate_source(article, row)


def test_cohort_and_original_predictions_unchanged():
    article, row = inputs()
    output = audit_sources([article], {"articles": [row], "originalHoldoutSha256": "frozen"})
    assert output["articleCount"] == 1
    assert output["priceDataAccessed"] is False
    assert output["originalPredictionsModified"] is False
    with pytest.raises(ValueError, match="ARTICLE_SET_MISMATCH"):
        audit_sources([article], {"articles": [{**row, "articleId": 2}]})
