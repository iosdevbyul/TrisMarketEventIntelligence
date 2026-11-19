"""Evidence and jurisdiction safeguards for historical event-price research.

Only explicit and auditable market mappings are allowed. Missing article bodies
are not evidence that an event had no price impact. These checks deliberately
favor abstention over a fabricated research observation.
"""
from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import Any

# Source provenance supplies jurisdiction, not evidence of economic causality.
KR_SOURCES = {"bok_corrections", "bok_statistics", "bok_press",
              "bok_releases", "bank_of_korea"}
KOREAN_BROAD_PROXY = "069500.KS"  # KODEX 200, a KR-listed liquid market proxy
KOREAN_SECTOR_PROXIES = {
    "FINANCIALS": "091170.KS",  # KODEX Bank, not an American finance ETF
}
EXPLICIT_ISSUERS = {
    "NVDA": ("nvidia", "엔비디아"),
    "AMD": ("amd", "advanced micro devices"),
    "000660.KS": ("sk hynix", "sk하이닉스", "sk 하이닉스"),
    "005930.KS": ("삼성전자", "samsung electronics"),
    "086790.KS": ("하나금융지주", "하나금융"),
}

TRADED_PROXIES = {
    "069500.KS": ("MARKET", "BROAD_MARKET"),
    "091170.KS": ("SECTOR", "FINANCIALS"),
}
OFFICIAL_DENIALS = (
    "사실과 다름", "사실이 아님", "사실이 아니",
    "검토하고 있지 않", "논의한 바 없", "추진하고 있지 않",
    "결정된 바 없", "결정된 내용이 없",
)
UNDECIDED_MARKERS = ("결정되지 않았", "확정되지 않았", "협의 중", "검토 중")


def event_fact_check(article: dict[str, Any]) -> dict[str, Any]:
    """Separate attributed media claims from the official response."""
    body = plain_text(article.get("summary"))
    title = plain_text(article.get("title"))
    correction = (str(article.get("source") or "") == "bok_corrections"
                  or "보도해명" in title)
    # The second section of a BOK correction is the authoritative response.
    # A denied claim quoted above the response is not the BOK's own policy.
    official_marker = re.search(
        r"(?:해명내용|한국은행(?: 및 기획재정부)? 입장|관계기관 입장|한은의 입장)"
        r"\s*[:：>]*", body,
    )
    response = body[official_marker.end():].strip() if official_marker else body
    has_denial = any(marker in response for marker in OFFICIAL_DENIALS)
    has_pending = any(marker in response for marker in UNDECIDED_MARKERS)
    status = (
        "OFFICIAL_DENIAL" if correction and has_denial else
        "NOT_FINALIZED" if correction and has_pending else
        "UNVERIFIED_CLARIFICATION" if correction else "UNKNOWN"
    )
    return {
        "sourceType": "OFFICIAL_CORRECTION" if correction else "OTHER",
        "officialStatementStatus": status,
        "reportedClaim": title,
        "officialResponseText": response[:1500] if correction else "",
        "officiallyConfirmedPolicyAction": False if correction else None,
        "limitation": ("A news allegation quoted in an official correction is "
                       "NOT a confirmed policy or an adopted decision."),
    }


def target_relevance(article: dict[str, Any], item: dict[str, Any]) -> str:
    """Exclude price claims without a recognizable economic transmission path."""
    text = (plain_text(article.get("title")) + " " +
            plain_text(article.get("summary"))).casefold()
    economic = (
        "금리", "통화정책", "물가", "환율", "외환", "스와프",
        "유동성", "대출", "채권", "주식",
        "기업", "수출", "수입", "경기", "성장률", "실적",
        "rate", "inflation", "earnings", "revenue",
        "semiconductor", "investment", "liquidity",
    )
    if not any(word in text for word in economic):
        return "NO_IDENTIFIABLE_MARKET_MECHANISM"
    return ""

MISSING_BODY_MARKERS = (
    "자세한 내용은 첨부파일", "자세한 내용은 붙임",
    "첨부파일을 참고", "첨부파일을 참조", "붙임 자료 참조",
    "첨부자료를 참고", "첨부자료를 참조",
    "please see attached", "refer to the attachment",
)


class _PlainText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def plain_text(value: Any) -> str:
    parser = _PlainText()
    parser.feed(html.unescape(str(value or "")))
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


def content_quality(article: dict[str, Any]) -> dict[str, Any]:
    """Require substantive, separately available body text for price attribution.

    A headline and an attachment pointer never constitute the event's actual
    statement. This remains conservative: a truly substantive short release
    may be deferred until a full source is acquired.
    """
    title = plain_text(article.get("title"))
    body = plain_text(article.get("summary"))
    lower = body.casefold()
    if not body or body.casefold() == title.casefold():
        reason = "TITLE_ONLY"
    elif any(marker in lower for marker in MISSING_BODY_MARKERS):
        reason = "ATTACHMENT_CONTENT_MISSING"
    elif len(body) < 45 or len(body.split()) < 6:
        reason = "INSUFFICIENT_ARTICLE_BODY"
    else:
        reason = ""
    return {"status": "ELIGIBLE" if not reason else "INSUFFICIENT",
            "reasonCode": reason, "bodyCharacters": len(body),
            "requiresSourceEnrichment": bool(reason)}


def article_market(article: dict[str, Any]) -> str:
    source = str(article.get("source") or "").lower()
    if source in KR_SOURCES or source.startswith("bok_"):
        return "KR"
    text = plain_text(article.get("title"))
    if "한국은행" in text or "금융통화위원회" in text:
        return "KR"
    return "UNKNOWN"


def localize_prediction(prediction: dict[str, Any],
                        article: dict[str, Any]) -> dict[str, Any]:
    """Do not score a Korean release against unrelated US sector/market ETFs.

    Equity STOCK mappings remain under the separate explicit-issuer validator;
    a genuine named cross-border stock can be kept. Unsupported regional
    sectors are rejected rather than silently mapped onto another country.
    """
    jurisdiction = article_market(article)
    result = dict(prediction)
    result["eventMarket"] = jurisdiction
    result["sourceContentQuality"] = content_quality(article)
    result["eventFacts"] = event_fact_check(article)
    if jurisdiction != "KR":
        return result
    allowed = []
    rejected = list(result.get("targetMappingRejections", []))
    for item in result.get("impacts", []):
        row = dict(item)
        kind = row.get("targetType")
        target = row.get("target")
        proxy = TRADED_PROXIES.get(str(row.get("priceProxy") or ""))
        if kind == "STOCK" and proxy is not None:
            kind, target = proxy
            row["targetType"] = kind
            row["target"] = target
        if kind == "MARKET" and target == "BROAD_MARKET":
            row["priceProxy"] = KOREAN_BROAD_PROXY
        elif kind == "SECTOR" and target in KOREAN_SECTOR_PROXIES:
            row["priceProxy"] = KOREAN_SECTOR_PROXIES[target]
        elif kind == "STOCK":
            ticker = str(row.get("priceProxy") or "")
            text = (plain_text(article.get("title")) + " " +
                    plain_text(article.get("summary"))).casefold()
            aliases = EXPLICIT_ISSUERS.get(ticker, ())
            if not (ticker.casefold() in text or
                    any(alias in text for alias in aliases)):
                rejected.append({"targetType": kind, "target": target,
                                 "reasonCode": "NO_VERIFIED_ARTICLE_ISSUER_LINK"})
                continue
        elif kind == "SECTOR":
            rejected.append({"targetType": kind, "target": target,
                             "reasonCode": "NO_VERIFIED_KR_SECTOR_PROXY"})
            continue
        elif kind != "STOCK":
            rejected.append({"targetType": kind, "target": target,
                             "reasonCode": "UNSUPPORTED_MARKET_TARGET"})
            continue
        relevance = target_relevance(article, row)
        if relevance:
            rejected.append({"targetType": kind, "target": target,
                             "reasonCode": relevance})
            continue
        allowed.append(row)
    result["impacts"] = allowed
    if rejected:
        result["targetMappingRejections"] = rejected
        result["recoveryStatus"] = (
            "PARTIAL_TARGET_MAPPING" if allowed else "NO_VALID_TARGETS"
        )
    return result
