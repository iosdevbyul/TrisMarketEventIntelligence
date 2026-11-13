"""Deterministic economic-event extraction; no model-generated facts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from html import unescape
from html.parser import HTMLParser

OFFICIAL_SOURCES = {"fed_monetary", "bok_monetary"}
EXTRACTOR_VERSION = "policy-v1"
RATE_CUES = re.compile(
    r"(?i)interest rates?|policy rates?|federal funds|target range|"
    r"기준금리|정책금리|금리 결정|금융통화위원회"
)
FORECAST_CUES = re.compile(
    r"(?i)\b(?:expect(?:ed|s)?|forecast|predict(?:ed|s)?|may|might|"
    r"could|would|likely|project(?:ed|s)?|anticipat(?:e|ed|es)|"
    r"proposal|proposed|rumou?r|speculat(?:e|ion)|plans? to)\b|"
    r"전망|예상|가능성|할 것으로|인상할|인하할|관측"
)
UP_CUES = re.compile(r"(?i)\b(?:raised?|increas(?:e|ed)|hiked?)\b|인상|올렸|상향")
DOWN_CUES = re.compile(r"(?i)\b(?:lower(?:ed)?|decreas(?:e|d)|cut|reduced?)\b|인하|내렸|하향")
HOLD_CUES = re.compile(r"(?i)\b(?:unchanged|maintain(?:ed)?|held|hold)\b|동결|유지")
DECISION_CUES = re.compile(
    r"(?i)\b(?:decided|voted|announced|has raised|has lowered|"
    r"raised|lowered|increased|decreased|cut|maintained|held)\b|"
    r"결정|발표|인상했|인하했|동결했|유지했"
)
BP = re.compile(r"(?i)(?<!\d)(\d{1,3}(?:\.\d+)?)\s*(?:basis points?|bps?)(?!\w)")
PERCENT_CHANGE = re.compile(
    r"(?i)(?<!\d)(\d{1,2}(?:\.\d+)?)\s*(?:percentage points?|%p|퍼센트포인트)"
)
KOREAN_BP = re.compile(r"(?<!\d)(\d{1,3}(?:\.\d+)?)\s*(?:베이시스포인트)")
MAX_TEXT = 12000


class _PlainText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def clean_text(value: str) -> str:
    parser = _PlainText()
    parser.feed(unescape(value[:MAX_TEXT]))
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()[:MAX_TEXT]


@dataclass(frozen=True)
class PolicyEvent:
    classification: str
    status: str
    direction: str | None
    change_basis_points: int | None
    evidence: str
    extractor_version: str = EXTRACTOR_VERSION


def _rate_change(text: str) -> int | None:
    matches: list[Decimal] = []
    for pattern, factor in ((BP, Decimal(1)), (KOREAN_BP, Decimal(1)),
                            (PERCENT_CHANGE, Decimal(100))):
        for match in pattern.finditer(text):
            try:
                matches.append(Decimal(match.group(1)) * factor)
            except InvalidOperation:
                continue
    if len(matches) != 1:
        return None  # Ambiguous or duplicated numeric assertions require review.
    value = matches[0]
    if value != value.to_integral_value() or not 1 <= value <= 1000:
        return None
    return int(value)


def extract_policy_event(source: str, title: str, summary: str | None) -> PolicyEvent | None:
    """Official feed provenance + unambiguous decision text is required for VERIFIED.

    RSS titles or excerpts are not assumed to be full decision documents.
    Candidates are deliberately not published as verified monetary decisions.
    """
    title_text = clean_text(title)
    summary_text = clean_text(summary or "")
    evidence = " ".join(v for v in (title_text, summary_text) if v)[:MAX_TEXT]
    if not RATE_CUES.search(evidence):
        return None
    if source not in OFFICIAL_SOURCES:
        return PolicyEvent("MONETARY_POLICY", "UNVERIFIED", None, None, evidence)
    # Detect speculation in either title or summary; conservatively hold for manual review.
    if FORECAST_CUES.search(evidence):
        return PolicyEvent("MONETARY_POLICY", "CANDIDATE", None, None, evidence)

    up, down, hold = (bool(p.search(evidence)) for p in (UP_CUES, DOWN_CUES, HOLD_CUES))
    directions = sum((up, down, hold))
    change = _rate_change(evidence)
    if directions != 1 or not DECISION_CUES.search(evidence):
        return PolicyEvent("MONETARY_POLICY", "CANDIDATE", None, None, evidence)
    if hold:
        # Holding rates requires explicit decision language but not numeric delta.
        return PolicyEvent("MONETARY_POLICY", "VERIFIED", "UNCHANGED", 0, evidence)
    if change is None:
        return PolicyEvent(
            "MONETARY_POLICY", "CANDIDATE", "UP" if up else "DOWN", None, evidence
        )
    return PolicyEvent(
        "MONETARY_POLICY", "VERIFIED", "UP" if up else "DOWN",
        change if up else -change, evidence,
    )
