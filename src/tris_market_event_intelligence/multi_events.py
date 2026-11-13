"""Conservative multi-category extraction from RSS titles and excerpts."""

import re
from dataclasses import dataclass

from tris_market_event_intelligence.events import (
    FORECAST_CUES,
    PolicyEvent,
    clean_text,
    extract_policy_event,
)

EXTRACTOR_VERSION = "multi-v2"

PATTERNS = (
    ("EARNINGS", "EARNINGS", r"(?i)\b(?:earnings|quarterly results|annual results|revenue report)\b|분기 실적|영업이익|어닝 서프라이즈|실적 발표"),
    ("CORPORATE_EVENT", "LEADERSHIP", r"(?i)\b(?:CEO|CFO|chief executive)\b.{0,55}\b(?:resigns?|steps? down|appointed|replaced|fired|departed|leaves?)\b|\b(?:resigns?|appointed|replaced)\b.{0,55}\b(?:CEO|CFO)\b|대표이사.{0,35}(?:사임|교체|선임)"),
    ("SERVICE_INCIDENT", "OUTAGE", r"(?i)\b(?:cloud|outlook|teams|azure|aws|service|network)\b.{0,55}\b(?:outage|disruption|down)\b|\b(?:outage|disruption)\b.{0,55}\b(?:cloud|service|network)\b|서비스 장애|클라우드 장애|접속 장애"),
    ("ECONOMIC_INDICATOR", "INFLATION", r"(?i)\b(?:consumer price index|producer price index|inflation rate|CPI|PPI)\b|소비자물가지수|생산자물가지수|물가상승률"),
    ("ECONOMIC_INDICATOR", "EMPLOYMENT", r"(?i)\b(?:nonfarm payrolls?|employment report|unemployment rate|jobless claims)\b|비농업 고용|실업률 발표|고용지표"),
    ("REGULATION", "TRADE_POLICY", r"(?i)\b(?:tariffs?|export controls?|import ban|sanctions?)\b|관세 부과|수출 규제|수입 금지|경제 제재|반도체 규제"),
    ("REGULATION", "FINANCIAL_POLICY", r"(?i)\b(?:securities regulation|antitrust ruling|delisting rule)\b|금융 규제|상장폐지 규정|자본시장 개혁|의무공개매수"),
    ("GEOPOLITICAL", "CONFLICT", r"(?i)\b(?:missile strike|military strikes?|armed conflict|ceasefire|war broke out|invasion)\b|군사 충돌|미사일 공격|휴전 합의|전쟁 발발|침공"),
    ("SUPPLY_CHAIN", "SEMICONDUCTOR", r"(?i)\b(?:chip|semiconductor|GPU|HBM)\b.{0,55}\b(?:supply|shortage|shipment|factory|production|export)\b|\b(?:supply|shortage|shipment)\b.{0,55}\b(?:chip|semiconductor|GPU|HBM)\b|(?:반도체|GPU).{0,40}(?:공급|생산|부족|수출)"),
    ("INDUSTRY", "AI_INFRASTRUCTURE", r"(?i)\b(?:AI infrastructure|data center investment|GPU deployment)\b|AI 인프라|데이터센터 투자|GPU 투자"),
    ("MARKET_MOVEMENT", "EQUITIES", r"(?i)\b(?:KOSPI|NASDAQ|Dow Jones)\b.{0,45}\b(?:plunges?|slumps?|surges?|rallies|falls?|drops?)\b|\b(?:plunges?|surges?|falls?|drops?)\b.{0,45}\b(?:KOSPI|NASDAQ)\b|(?:코스피|나스닥|코스닥).{0,40}(?:급락|급등|하락|상승)"),
)

# Additional expressions prompted by real, source-linked 2026-11-13 headlines.
# They remain conservative lexical cues, not factual confirmation.
ADDITIONAL_PATTERNS = (
    ("CORPORATE_EVENT", "LEADERSHIP",
     r"(?i)\b(?:CEO|chief executive)\s+(?:transition|succession)\b"),
    ("REGULATION", "FINANCIAL_POLICY",
     r"(?i)\bcapital markets? reforms?\b"),
    ("REGULATION", "MAP_DATA_POLICY",
     r"(?i)\b(?:map data exports?|high-precision map data|"
     r"security requirements for high-precision map data)\b"),
    ("SERVICE_INCIDENT", "OUTAGE",
     r"(?i)\b(?:Microsoft 365|Office 365)\b.{0,55}\b(?:outage|disruption)\b"),
    ("GEOPOLITICAL", "CONFLICT",
     r"(?i)\b(?:Iranian attack drones?|strike on (?:a |the )?(?:US|U\.S\.) radar)\b"),
    ("MARKET_MOVEMENT", "EQUITIES",
     r"(?i)\bKOSPI\b.{0,65}\b(?:surrenders?|slides?|reverses?)\b|"
     r"\btech stocks\b.{0,55}\b(?:taking a beating|selloff|slide|fall|drop)\b"),
    ("SUPPLY_CHAIN", "SEMICONDUCTOR",
     r"(?i)\b(?:supply|shipment|pledge)\b.{0,65}\b(?:GPUs?|semiconductors?|chips?)\b"),
)

RUMOR = re.compile(r"(?i)\b(?:rumou?r(?:ed|s)?|unconfirmed|alleged)\b|루머|미확인")
ATTRIBUTED = re.compile(r"(?i)\b(?:reportedly|according to sources|sources say)\b|관계자에 따르면")
OFFICIAL_CATEGORIES = {
    "bls_cpi": {"ECONOMIC_INDICATOR"},
    "bls_ppi": {"ECONOMIC_INDICATOR"},
    "bls_employment": {"ECONOMIC_INDICATOR"},
    "sec_press": {"REGULATION", "CORPORATE_EVENT"},
    "fed_all": {"MONETARY_POLICY", "REGULATION"},
}


@dataclass(frozen=True)
class ClassifiedEvent:
    event: PolicyEvent
    claim_kind: str
    tags: tuple[str, ...]
    verification_note: str


def extract_news_event(
    source: str, title: str, summary: str | None
) -> ClassifiedEvent | None:
    evidence = " ".join(
        part for part in (clean_text(title), clean_text(summary or "")) if part
    )
    if not evidence:
        return None
    claim = "REPORTED"
    if RUMOR.search(evidence):
        claim = "RUMOR"
    elif FORECAST_CUES.search(evidence):
        claim = "FORECAST"
    elif ATTRIBUTED.search(evidence):
        claim = "ATTRIBUTED_REPORT"

    policy = extract_policy_event(source, title, summary)
    if policy is not None and source in {"fed_monetary", "bok_monetary"}:
        status = policy.status if claim == "REPORTED" else "CANDIDATE"
        event = PolicyEvent(
            "MONETARY_POLICY", status,
            policy.direction if status == "VERIFIED" else None,
            policy.change_basis_points if status == "VERIFIED" else None,
            policy.evidence, EXTRACTOR_VERSION,
        )
        return ClassifiedEvent(
            event, claim, ("MONETARY_POLICY",),
            "RULE_SUPPORTED_OFFICIAL_FEED" if status == "VERIFIED" else "REVIEW_REQUIRED",
        )

    matches = [
        (category, tag) for category, tag, regex in PATTERNS + ADDITIONAL_PATTERNS
        if re.search(regex, evidence)
    ]
    if policy is not None:
        matches.append(("MONETARY_POLICY", "MONETARY_POLICY"))
    if not matches:
        return None
    category = matches[0][0]
    tags = tuple(dict.fromkeys(tag for _, tag in matches))[:8]
    official = category in OFFICIAL_CATEGORIES.get(source, set())
    status = "CANDIDATE" if official else "UNVERIFIED"
    return ClassifiedEvent(
        PolicyEvent(category, status, None, None, evidence, EXTRACTOR_VERSION),
        claim, tags, "PRIMARY_DOCUMENT_NOT_CHECKED" if official else "THIRD_PARTY_REPORT",
    )
