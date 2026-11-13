"""Article-level, multi-claim discovery without LLM inference or factual certification."""

from __future__ import annotations

import re
from dataclasses import dataclass

from tris_market_event_intelligence.events import FORECAST_CUES, clean_text, extract_policy_event
from tris_market_event_intelligence.multi_events import (
    ADDITIONAL_PATTERNS,
    ATTRIBUTED,
    OFFICIAL_CATEGORIES,
    PATTERNS,
    RUMOR,
)

CLAIM_VERSION = "claims-v1"
MAX_CLAIMS_PER_ARTICLE = 24
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+|\s*;\s*")
DENIAL = re.compile(
    r"(?i)\b(?:den(?:y|ies|ied)|reject(?:s|ed)?|refut(?:e|es|ed)|"
    r"no evidence of|did not|has not|never|false report)\b|"
    r"부인|사실무근|없다고|아니다|않았|거짓"
)
PAST = re.compile(
    r"(?i)\b(?:last year|last month|in 20\d{2}|previously|historical|"
    r"years ago|was reported last)\b|지난해|작년에|과거|지난달|전년도"
)
POLICY_SOURCE = frozenset({"fed_monetary", "bok_monetary"})


@dataclass(frozen=True)
class EventClaim:
    classification: str
    tag: str
    claim_kind: str
    polarity: str
    temporal_context: str
    status: str
    evidence: str
    verification_note: str


def _clauses(title: str, summary: str | None) -> list[str]:
    # Treat headlines and description as separate evidence spans; never borrow
    # certainty or negation from an unrelated clause.
    result: list[str] = []
    for text in (title, summary or ""):
        cleaned = clean_text(text)
        if cleaned:
            result.extend(x.strip() for x in SENTENCE_BOUNDARY.split(cleaned) if x.strip())
    return list(dict.fromkeys(result))


def _claim_kind(text: str, polarity: str, temporal: str) -> str:
    if RUMOR.search(text):
        return "RUMOR"
    if FORECAST_CUES.search(text):
        return "FORECAST"
    if ATTRIBUTED.search(text):
        return "ATTRIBUTED_REPORT"
    if polarity == "DENIED":
        return "DENIAL"
    if temporal == "PAST":
        return "HISTORICAL_REFERENCE"
    return "REPORTED"


def extract_claims(source: str, title: str, summary: str | None) -> list[EventClaim]:
    """Return separate typed claims per evidence clause; never merge unrelated clauses.

    Output is only *news claims*. A positive classification is not evidence
    that an underlying real-world event happened.
    """
    claims: list[EventClaim] = []
    seen: set[tuple[str, str, str]] = set()
    for clause in _clauses(title, summary):
        polarity = "DENIED" if DENIAL.search(clause) else "AFFIRMED"
        temporal = "PAST" if PAST.search(clause) else "UNSPECIFIED"
        kind = _claim_kind(clause, polarity, temporal)
        matched = [
            (category, tag) for category, tag, pattern in PATTERNS + ADDITIONAL_PATTERNS
            if re.search(pattern, clause)
        ]
        policy = extract_policy_event(source, clause, None)
        if re.search(
            r"(?i)\b(?:CEO|CFO|chief executive)\b.{0,55}"
            r"\b(?:resignation|resigned|resigns|resign|departure)\b"
            r"|대표이사.{0,35}(?:사임|퇴임|교체)",
            clause,
        ):
            matched.append(("CORPORATE_EVENT", "LEADERSHIP"))
        if policy is not None:
            matched.append(("MONETARY_POLICY", "MONETARY_POLICY"))
        for classification, tag in dict.fromkeys(matched):
            key = (classification, tag, clause.casefold())
            if key in seen:
                continue
            seen.add(key)
            official = (
                classification in OFFICIAL_CATEGORIES.get(source, set())
                or (classification == "MONETARY_POLICY" and source in POLICY_SOURCE)
            )
            # A mere source reputation never verifies a non-policy event.
            rule_verified = (
                classification == "MONETARY_POLICY"
                and source in POLICY_SOURCE
                and policy is not None
                and policy.status == "VERIFIED"
                and kind == "REPORTED"
                and polarity == "AFFIRMED"
                and temporal != "PAST"
            )
            if rule_verified:
                status, note = "VERIFIED", "RULE_SUPPORTED_OFFICIAL_FEED"
            elif official:
                status, note = "CANDIDATE", "PRIMARY_DOCUMENT_NOT_CHECKED"
            else:
                status, note = "UNVERIFIED", "THIRD_PARTY_REPORT"
            claims.append(EventClaim(
                classification, tag, kind, polarity, temporal, status,
                clause, note,
            ))
            if len(claims) >= MAX_CLAIMS_PER_ARTICLE:
                return claims
    return claims
