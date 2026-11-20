"""Predeclared economic-event research eligibility independent of future prices.

This is a conservative text/provenance screen for *candidate* events, not a
claim that a headline caused an actual asset return or that a ticker mapping
has already been validated. All criteria use article data available to the
selector; retrospective PDF text remains explicitly non-PIT-certified.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

from tris_market_event_intelligence.event_evidence_policy import (
    content_quality,
    plain_text,
)
from tris_market_event_intelligence.event_grounding import event_facts

# Concrete economic releases/decisions, not general investment commentary.
CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("MONETARY_POLICY_DECISION", (
        "통화정책방향", "기준금리 결정", "기준금리를", "금융통화위원회",
        "interest rate decision", "central bank cuts rates",
        "central bank raises rates", "fed cuts rates", "fed raises rates",
    )),
    ("ECONOMIC_INDICATOR_RELEASE", (
        "국제수지", "국내총생산", "소비자물가", "생산자물가",
        "수출입물가지수", "무역지수", "금융기관 가중평균금리",
        "가계신용", "경기조사 결과", "경제심리지수", "고용동향",
        "산업생산", "소매판매", "통화 및 유동성",
        "inflation data", "gdp growth", "employment report",
        "unemployment rate", "consumer price index", "trade deficit",
    )),
    ("EARNINGS_OR_GUIDANCE", (
        "실적 발표", "실적 전망", "영업이익", "매출액",
        "earnings report", "quarterly results", "profit warning",
        "raises guidance", "cuts guidance", "revenue forecast",
        "earnings outlook", "quarterly profit",
    )),
    ("CORPORATE_ACTION_OR_CONTRACT", (
        "인수 계약", "합병 계약", "공급계약", "수주 계약", "신규 투자",
        "acquisition agreement", "merger announced", "supply contract",
        "wins contract", "factory investment", "capital expenditure",
    )),
    ("REGULATORY_ACTION", (
        "규제 시행", "법안 통과", "관세 부과", "수출 제한",
        "tariffs imposed", "export restrictions", "regulator approves",
        "regulator blocks", "sanctions announced",
    )),
)
NON_EVENT_HEADLINES = (
    "tech life", "tech now", "why are", "why gen", "how do",
    "how a", "will the threat", "explained", "opinion",
    "podcast", "newsletter", "review:", "watch:", "listen:",
)
NON_EVENT_SOURCES = ("bbc_technology_program",)
NON_ACTION_STATEMENTS = (
    "금융통화신용정책보고서", "통화신용정책보고서",
    "기업경영분석 결과", "통화 및 유동성 동향",
)


def research_candidate(article: dict[str, Any]) -> dict[str, Any]:
    """Screen a market event without using model output or realized returns."""
    title = plain_text(article.get("title"))
    body = plain_text(article.get("summary"))
    title_lower = title.casefold()
    text = (title + " " + body).casefold()
    source = str(article.get("source") or "")
    quality = content_quality(article)
    facts = event_facts(article)
    reasons = []
    if quality["status"] != "ELIGIBLE":
        reasons.append("INSUFFICIENT_OR_PLACEHOLDER_BODY")
    if not article.get("sourceUrl"):
        reasons.append("SOURCE_URL_MISSING")
    if title_lower.startswith(NON_EVENT_HEADLINES):
        reasons.append("EDITORIAL_OR_PROGRAMME_NOT_EVENT")
    if source in NON_EVENT_SOURCES:
        reasons.append("NON_EVENT_MEDIA_FORMAT")
    if facts["sourceType"] == "OFFICIAL_CORRECTION":
        reasons.append("OFFICIAL_CORRECTION_NOT_NEW_CONFIRMED_ACTION")
    matched = [
        category for category, phrases in CATEGORIES
        if any(phrase in text for phrase in phrases)
    ]
    # SEC 8-K item metadata is official evidence that a filing reports
    # results (2.02) or a material agreement (1.01), but does not prove
    # the first external announcement time or actual economic surprise.
    if source.startswith("sec_8k_") and article.get("sourceUrl", "").startswith(
        "https://www.sec.gov/Archives/edgar/data/"
    ):
        if "item 2.02" in title_lower and "EARNINGS_OR_GUIDANCE" not in matched:
            matched.insert(0, "EARNINGS_OR_GUIDANCE")
        elif "item 1.01" in title_lower and (
            "CORPORATE_ACTION_OR_CONTRACT" not in matched
        ):
            matched.insert(0, "CORPORATE_ACTION_OR_CONTRACT")
    # A board review of financial stability is NOT a monetary-policy
    # decision. M2 and retail bank rate statistics measure conditions but
    # do not by themselves announce a central-bank policy change.
    if "금융안정 상황" in title_lower or "금융안정회의" in title_lower:
        matched = ["FINANCIAL_STABILITY_REVIEW"]
    elif any(key in title_lower for key in (
        "통화 및 유동성", "금융기관 가중평균금리",
    )):
        matched = ["ECONOMIC_INDICATOR_RELEASE"]
    elif source.startswith("sec_8k_"):
        if "item 1.01" in title_lower:
            matched = ["CORPORATE_ACTION_OR_CONTRACT"]
        elif "item 2.02" in title_lower:
            matched = ["EARNINGS_OR_GUIDANCE"]
    if not matched:
        reasons.append("NO_SPECIFIC_ECONOMIC_EVENT")
    if any(value in title_lower for value in NON_ACTION_STATEMENTS):
        reasons.append("GENERAL_REPORT_NOT_DIRECT_MARKET_EVENT")
    if re.search(r"\b(?:could|might|may|would|should)\b", title_lower) and not (
        "EARNINGS_OR_GUIDANCE" in matched or "MONETARY_POLICY_DECISION" in matched
    ):
        reasons.append("SPECULATIVE_HEADLINE")
    evidence = article.get("contentEvidence")
    return {
        "eligible": not reasons,
        "eventCategory": matched[0] if matched else None,
        "matchedCategories": matched,
        "reasons": sorted(set(reasons)),
        "bodyQuality": quality["status"],
        "officialDenial": facts["officialDenial"],
        "sourceDocumentAvailable": bool(evidence),
        "originalContentPITCertified": bool(
            isinstance(evidence, dict) and evidence.get("pointInTimeCertified") is True
        ),
        "sourceType": source,
        "qualification": "PRE_EVENT_TEXT_POLICY_NOT_PRICE_CAUSALITY",
    }


def filter_market_event_candidates(
    articles: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    included = []
    reason_counts: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    for article in articles:
        report = research_candidate(article)
        if report["eligible"]:
            included.append(article)
            by_category[str(report["eventCategory"])] += 1
        else:
            reason_counts.update(report["reasons"])
    return included, {
        "selectionRule": "explicit_economic_event_provenance_v1",
        "evaluatedCandidateArticles": len(articles),
        "economicEventCandidates": len(included),
        "rejectedCandidateArticles": len(articles) - len(included),
        "rejectionReasons": dict(sorted(reason_counts.items())),
        "eventCategoryCounts": dict(sorted(by_category.items())),
        "selectionUsesRealizedMarketReturns": False,
        "causalRelevanceProven": False,
    }
