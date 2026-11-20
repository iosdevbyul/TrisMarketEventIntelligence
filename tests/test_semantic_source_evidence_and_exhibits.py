"""Regression for 30-case semantic failure diagnoses and official SEC exhibits."""
from __future__ import annotations

import json

import httpx
import pytest

from tris_market_event_intelligence.market_event_candidate_policy import (
    research_candidate,
)
from tris_market_event_intelligence.sec_filing_exhibit_evidence import (
    accession_archive,
    enrich_sec_article,
    exhibit_urls,
)
from tris_market_event_intelligence.semantic_source_evidence import (
    expand_evidence_ids,
    source_excerpts,
)
from tris_market_event_intelligence.source_grounded_semantic_analysis import (
    validate_semantic_interpretation,
)
from tris_market_event_intelligence.source_grounded_semantic_replay_cli import (
    semantic_replay,
)


def sec_article() -> dict:
    return {
        "articleId": 20873, "source": "sec_8k_2488",
        "title": "ADVANCED MICRO DEVICES INC SEC 8-K Item 2.02 EARNINGS_8K",
        "sourceUrl": (
            "https://www.sec.gov/Archives/edgar/data/2488/"
            "000000248826000072/amd-20260505.htm"
        ),
        "publishedAtReported": "2026-05-05T20:16:06+00:00",
        "summary": (
            "On May 5, 2026, AMD announced its financial results for Q1 2026. "
            "A copy of the detailed results is attached as Exhibit 99.1. "
            "The attached financial information was provided to investors."
        ),
    }


def test_only_preexisting_evidence_id_can_supply_factual_quote():
    article = sec_article()
    excerpts = source_excerpts(article["summary"])
    assert excerpts
    answer = {
        "eventCategory": "EARNINGS_OR_GUIDANCE",
        "eventDescription": "Company disclosed earnings",
        "sourceStatements": [
            {"evidenceId": excerpts[0]["evidenceId"], "factualStatus": "OFFICIAL",
             "interpretation": "Results announced"},
        ],
        "economicMechanisms": [
            {"evidenceId": excerpts[0]["evidenceId"], "affectedScope": "ISSUER",
             "economicDirection": "UNKNOWN", "reasoning": "Hypothesis: result affects issuer",
             "uncertainty": "No numerical earnings figures available"},
        ],
        "unverifiedAssumptions": [], "missingEvidence": ["EX-99.1 not captured"],
        "overallEconomicDirection": "UNKNOWN",
    }
    result = expand_evidence_ids(answer, excerpts)
    certified = validate_semantic_interpretation(article, result)
    assert certified["modelAnalysis"]["sourceStatements"][0]["quote"] in article["summary"]
    assert certified["economicTargetLinks"][0]["target"] == "AMD"
    answer["sourceStatements"][0]["evidenceId"] = "E999"
    with pytest.raises(ValueError, match="SEMANTIC_EVIDENCE_ID_NOT_IN_SOURCE"):
        expand_evidence_ids(answer, excerpts)


def test_excerpt_selection_is_bounded_deterministic_and_source_only():
    body = " ".join(
        f"Paragraph {i} reports operating revenue of {i} billion."
        for i in range(250)
    )
    first = source_excerpts(body)
    assert len(first) <= 26
    assert first == source_excerpts(body)
    assert all(row["text"] in body for row in first)
    assert all(len(row["text"]) <= 260 for row in first)


def test_repeated_money_supply_and_bank_rates_are_not_policy_decisions():
    for title in (
        "2026년 4월 통화 및 유동성",
        "2026년 8월 금융기관 가중평균금리",
    ):
        article = {
            "title": title, "source": "bok_all",
            "sourceUrl": "https://www.bok.or.kr/example",
            "summary": (
                "금융통화위원회 관련 뉴스와 M2 통화량 증가가 보고됐으며 "
                "금융기관 대출금리와 저축성 수신금리의 변화가 발표됐다."
            ),
        }
        result = research_candidate(article)
        assert result["eventCategory"] == "ECONOMIC_INDICATOR_RELEASE"
    stability = {
        "title": "「금융안정 상황」(2026년 9월)",
        "source": "bok_all",
        "sourceUrl": "https://www.bok.or.kr/example",
        "summary": (
            "한국은행은 금융통화위원회 정기회의에서 금융안정 상황을 "
            "점검하였으며 정기 조사 결과를 발표했다."
        ),
    }
    assert research_candidate(stability)["eventCategory"] == "FINANCIAL_STABILITY_REVIEW"


def test_official_sec_index_exhibit_urls_reject_other_ciks_and_hosts():
    article = sec_article()
    base, idx = accession_archive(article)
    assert idx.endswith("0000002488-26-000072-index.htm")
    raw = (
        b"<table>"
        b'<tr><td><a href="/Archives/edgar/data/2488/000000248826000072/'
        b'amd-q1-release.htm">release</a></td><td>EX-99.1</td></tr>'
        b'<tr><td><a href="https://evil.example.net/ex99.2.htm">'
        b'wrong</a></td><td>EX-99.2</td></tr>'
        b'<tr><td><a href="/Archives/edgar/data/2488/000000248826000072/'
        b'amd-slide.htm">slides</a></td><td>EX-99.2</td></tr>'
        b"</table>"
    )
    links = exhibit_urls(raw, article)
    assert links == {
        "99.1": base + "amd-q1-release.htm",
        "99.2": base + "amd-slide.htm",
    }
    with pytest.raises(ValueError, match="SEC_EXHIBIT_ARCHIVE_PROVENANCE_MISMATCH"):
        accession_archive({
            **article, "sourceUrl": "https://evil.example.net/filing.htm",
        })


def test_real_exhibit_is_added_as_source_with_verifiable_sha_not_inferred_earnings():
    article = sec_article()
    base, index_url = accession_archive(article)
    index_html = (
        b"<table><tr><td><a href=\"amd-results.htm\">"
        b"Earnings press release</a></td><td>EX-99.1</td></tr></table>"
    )
    actual_exhibit = (
        b"<html><body><h2>Quarterly Results</h2>"
        b"<p>AMD reported quarterly revenue of $8.2 billion and "
        b"net income of $1.4 billion for the quarter.</p>"
        b"</body></html>"
    )
    requested = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        if str(request.url) == index_url:
            return httpx.Response(200, content=index_html)
        if str(request.url) == base + "amd-results.htm":
            return httpx.Response(200, content=actual_exhibit)
        raise AssertionError("Unexpected SEC source URL")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        enriched, metadata = enrich_sec_article(
            article, client=client, user_agent="Research real@example.com",
        )
    assert len(requested) == 2
    assert metadata["status"] == "PARTIAL_EXHIBITS_ACQUIRED"
    assert metadata["missingExhibits"] == ["99.2"]
    assert metadata["firstPublicationPITCertified"] is False
    assert "revenue of $8.2 billion" in enriched["summary"]
    assert "SEC EXHIBIT 99.1 SOURCE TEXT" in enriched["summary"]
    assert "revenue of $8.2 billion" not in article["summary"]
    import hashlib

    assert metadata["acquiredExhibits"][0]["rawSha256"] == hashlib.sha256(
        actual_exhibit,
    ).hexdigest()


def test_replay_handles_invalid_model_evidence_reference_without_silent_success(
    monkeypatch, tmp_path,
):
    from tris_market_event_intelligence import (
        source_grounded_semantic_replay_cli as replay,
    )
    article = sec_article()
    monkeypatch.setattr(replay, "source_items", lambda *_: (
        [article], "same-original-report", {"inventoryAsOf": "2026-11-20T03:55:45+00:00"},
    ))

    class BadModel:
        def generate_json(self, *, system, user, schema):
            payload = json.loads(user)
            assert "sourceExcerpts" in payload
            assert "sourceBody" not in payload
            assert "actualDirection" not in payload
            return json.dumps({
                "eventCategory": "EARNINGS_OR_GUIDANCE",
                "eventDescription": "Company reported earnings",
                "sourceStatements": [
                    {"evidenceId": "FAKE-EXCERPT", "factualStatus": "OFFICIAL",
                     "interpretation": "Imagined result"},
                ],
                "economicMechanisms": [],
                "unverifiedAssumptions": [], "missingEvidence": [],
                "overallEconomicDirection": "UNKNOWN",
            })

    output = tmp_path / "semantic-v2.json"
    result = semantic_replay(
        holdout_result=tmp_path / "original.json",
        model=BadModel(), output=output,
    )
    assert result["summary"]["statusCounts"] == {"INVALID_MODEL_ANALYSIS": 1}
    report = json.loads(output.read_text())
    assert report["articles"][0]["errorCode"] == "SEMANTIC_EVIDENCE_ID_NOT_IN_SOURCE"
    assert report["priceDataAccessed"] is False


def test_contract_exhibit_10_1_is_officially_selected_without_other_hosts():
    article = {
        **sec_article(),
        "title": "AMD SEC 8-K Item 1.01 MATERIAL_CONTRACT_8K",
    }
    base, _ = accession_archive(article)
    filing_index = (
        b'<table><tr><td><a href="amd-agreement.htm">'
        b'Material definitive agreement</a></td><td>EX-10.1</td></tr>'
        b'<tr><td><a href="https://malicious.example/contract.htm">'
        b'unknown</a></td><td>EX-99.1</td></tr></table>'
    )
    urls = exhibit_urls(filing_index, article)
    assert urls == {"10.1": base + "amd-agreement.htm"}


def test_truncated_model_attempt_retries_with_reduced_evidence_context(
    monkeypatch, tmp_path,
):
    from tris_market_event_intelligence import (
        source_grounded_semantic_replay_cli as replay,
    )
    article = sec_article()
    article["summary"] = " ".join(
        f"The company reported quarterly revenue of {i} million dollars. "
        for i in range(130)
    )
    monkeypatch.setattr(replay, "source_items", lambda *_: (
        [article], "untouched-study", {"inventoryAsOf": "2026-11-20T03:55:45+00:00"},
    ))
    lengths = []

    class TruncatingModel:
        def generate_json(self, *, system, user, schema):
            content = json.loads(user)
            lengths.append(len(content["sourceExcerpts"]))
            if len(lengths) == 1:
                raise RuntimeError("MODEL_GENERATION_TRUNCATED")
            first = content["sourceExcerpts"][0]["evidenceId"]
            return json.dumps({
                "eventCategory": "EARNINGS_OR_GUIDANCE",
                "eventDescription": "Official issuer quarterly disclosure",
                "sourceStatements": [
                    {"evidenceId": first, "factualStatus": "OFFICIAL",
                     "interpretation": "Official quarterly revenue released"},
                ],
                "economicMechanisms": [
                    {"evidenceId": first, "affectedScope": "ISSUER",
                     "economicDirection": "UNKNOWN",
                     "reasoning": "No revenue consensus; hypothesis only",
                     "uncertainty": "Cannot infer price direction"},
                ],
                "unverifiedAssumptions": [], "missingEvidence": [],
                "overallEconomicDirection": "UNKNOWN",
            })

    report = semantic_replay(
        holdout_result=tmp_path / "untouched-original.json",
        model=TruncatingModel(), output=tmp_path / "immutable-new.json",
    )
    assert len(lengths) == 2
    assert lengths[0] > 12
    assert lengths[1] == 12
    assert report["summary"]["quoteAndSchemaValidated"] == 1
    assert report["summary"]["statusCounts"]["COMPLETE"] == 1
