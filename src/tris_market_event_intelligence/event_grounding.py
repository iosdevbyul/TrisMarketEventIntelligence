"""Event-level evidence contract for market research.

This module does not infer causality from prices. It separates the alleged
press narrative, official denial, and whether the article supplies evidence
of an actionable market impact at all.
"""
from __future__ import annotations

import re
from typing import Any

from tris_market_event_intelligence.event_evidence_policy import plain_text

DENIALS = (
    "사실이 아님", "사실과 다름", "사실이 아니",
    "검토하고 있지 않", "논의한 바 없", "추진하고 있지 않",
    "결정된 바 없", "확정된 바 없", "보도내용은 사실",
    "reported claim is false", "not considering",
)
CLAIM_LABELS = ("보도내용", "언론 보도내용", "언론보도", "언론 보도", "보도 내용")
OFFICIAL_LABELS = (
    "해명내용", "한국은행 입장", "한국은행의 입장", "관계기관 입장",
    "한국은행 및 기획재정부 입장", "해명 내용", "공식 입장",
)
NON_MARKET = (
    "기념주화", "청첩장", "급여후생", "노조 집행부",
    "총재 연봉", "공동숙소비", "2만원권 발행",
)
BOILERPLATE = {
    "http", "https", "tel", "www", "fax", "공보실", "문의처",
    "보도내용", "해명내용", "보도해명자료", "관련하여", "바랍니다",
    "알려드립니다", "다음과", "해명합니다", "첨부파일",
    "보도와", "보도에", "사실이", "참고해주시기", "한국은행",
    "내용은", "있음을", "자료를", "배포시부터",
}


def _section(text: str, labels: tuple[str, ...]) -> int:
    starts = [text.find(label) for label in labels if label in text]
    return min(starts) if starts else -1


def event_facts(article: dict[str, Any]) -> dict[str, Any]:
    """Parse explicit official denials without asserting the rumor was true."""
    title = plain_text(article.get("title"))
    body = plain_text(article.get("summary"))
    source = str(article.get("source") or "").lower()
    official = source == "bok_corrections" or "보도해명" in title
    denial = official and any(word in body.casefold() for word in DENIALS)
    claim_at = _section(body, CLAIM_LABELS)
    official_at = _section(body, OFFICIAL_LABELS)
    press_claim = body[claim_at:official_at].strip() if (
        claim_at >= 0 and official_at > claim_at
    ) else title
    official_statement = body[official_at:].strip() if official_at >= 0 else body
    immaterial = any(word in (title + " " + body).casefold() for word in NON_MARKET)
    return {
        "sourceType": "OFFICIAL_CORRECTION" if official else "NEWS",
        "pressClaim": press_claim[:1400],
        "officialStatement": official_statement[:1800] if official else "",
        "officialDenial": denial,
        "confirmedPolicy": False if denial else None,
        "nonMarketAdministrative": immaterial,
        "evidenceMode": "RETROSPECTIVE_SOURCE_NOT_PIT_CERTIFIED",
    }


def is_denial_inversion(prediction: dict[str, Any], facts: dict[str, Any]) -> bool:
    """Block a summary that positively ATTRIBUTES a disavowed claim to BOK.

    We avoid broad language-model heuristics; only clear assertion verbs
    combined with the official denial are caught here.
    """
    if not facts["officialDenial"]:
        return False
    summary = plain_text(prediction.get("summary")).casefold()
    return bool(
        re.search(
            r"(?:confirmed|announced|implemented|approved|decided|finalized)"
            r".{0,70}(?:policy|plan|measure|swap|rate|investment|intervention)",
            summary,
        ) and not any(term in summary for term in (
            "denied", "not ", "false", "incorrect", "untrue", "refuted",
            "no decision",
        ))
    )


def enforce_event_grounding(
    prediction: dict[str, Any], article: dict[str, Any],
) -> dict[str, Any]:
    facts = event_facts(article)
    result = dict(prediction)
    result["eventFacts"] = facts
    rejection = []
    if is_denial_inversion(prediction, facts):
        rejection.append("OFFICIAL_DENIAL_REVERSED")
    if facts["nonMarketAdministrative"]:
        rejection.append("NO_DIRECT_ECONOMIC_EXPOSURE")
    if rejection:
        result["eventGroundingStatus"] = "ABSTAIN"
        result["eventGroundingReasons"] = rejection
        result["impacts"] = []
    else:
        result["eventGroundingStatus"] = "PROVISIONAL"
    return result


def substantive_terms(article: dict[str, Any]) -> set[str]:
    body = (plain_text(article.get("title")) + " " +
            plain_text(article.get("summary"))).casefold()
    # Exclude links, contacts and boilerplate words which are not event keys.
    return {
        token for token in re.findall(r"[a-zA-Z가-힣]{3,}", body)
        if token not in BOILERPLATE and not token.startswith("http")
    }
