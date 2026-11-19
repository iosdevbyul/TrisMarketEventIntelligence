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
    if jurisdiction != "KR":
        return result
    allowed = []
    rejected = list(result.get("targetMappingRejections", []))
    for item in result.get("impacts", []):
        row = dict(item)
        kind = row.get("targetType")
        target = row.get("target")
        if kind == "MARKET" and target == "BROAD_MARKET":
            row["priceProxy"] = KOREAN_BROAD_PROXY
        elif kind == "SECTOR" and target in KOREAN_SECTOR_PROXIES:
            row["priceProxy"] = KOREAN_SECTOR_PROXIES[target]
        elif kind == "SECTOR":
            rejected.append({"targetType": kind, "target": target,
                             "reasonCode": "NO_VERIFIED_KR_SECTOR_PROXY"})
            continue
        elif kind != "STOCK":
            rejected.append({"targetType": kind, "target": target,
                             "reasonCode": "UNSUPPORTED_MARKET_TARGET"})
            continue
        allowed.append(row)
    result["impacts"] = allowed
    if rejected:
        result["targetMappingRejections"] = rejected
        result["recoveryStatus"] = (
            "PARTIAL_TARGET_MAPPING" if allowed else "NO_VALID_TARGETS"
        )
    return result
