"""Compact Korean/English article evidence before small local-model inference.

Preserve verbatim excerpts. Prefer named claims and prose over numeric tables;
do not manufacture an interpretation or a market conclusion.
"""
from __future__ import annotations

import re

STATEMENT_START = re.compile(r"(?=□|(?<!\w)[―￭])")
CLAIM_VERBS = re.compile(
    r"증가|감소|상승|하락|확대|축소|개선|악화|성장|둔화|전망|기록|발표|"
    r"유지|계약|체결|판매|투자|수출|수입|인상|인하|흑자|적자|"
    r"rose|fell|grew|declined|increased|decreased|revenue|profit|loss|"
    r"agreement|contract|signed|guidance|forecast", re.IGNORECASE
)
LETTER = re.compile(r"[가-힣A-Za-z]")
NUMBER = re.compile(r"\d")


def compact_source(text: str, title: str, *, max_chars: int = 6500) -> str:
    """Retain article lead plus substantive excerpts within a predictable budget."""
    marker = " OFFICIAL BOK ATTACHMENT TEXT: "
    if marker in text:
        original, attachment = text.split(marker, 1)
        lead = original[:min(1200, max_chars // 3)]
        remaining = attachment
    else:
        if len(text) <= max_chars:
            return text
        lead = text[:min(1200, max_chars // 3)]
        remaining = text[len(lead):]
    spans = STATEMENT_START.split(remaining)
    if len(spans) == 1:
        spans = re.split(r"(?<=[.!?。])\s+", remaining)
    title_terms = [x.lower() for x in re.findall(r"[가-힣]{2,}|[A-Za-z]{4,}", title)]
    candidates: list[tuple[float, int, str]] = []
    for index, raw in enumerate(spans):
        piece = " ".join(raw.split())
        if len(piece) < 26:
            continue
        # Do not dump all spreadsheet values into the model context.
        piece = piece[:620]
        letters = len(LETTER.findall(piece))
        numbers = len(NUMBER.findall(piece))
        if letters < 14 or numbers > letters * 1.2:
            continue
        score = 3.0 if CLAIM_VERBS.search(piece) else 0.0
        score += sum(1.5 for term in title_terms if term in piece.lower())
        score += 1.5 if ("□" in piece or "―" in piece) else 0.0
        score -= max(0, numbers - letters * 0.6) / 25
        if score > 0:
            candidates.append((score, index, piece))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected = []
    remaining_chars = max_chars - len(lead) - 2
    for _, _, candidate in candidates:
        if len(candidate) + 2 > remaining_chars:
            continue
        selected.append(candidate)
        remaining_chars -= len(candidate) + 2
        if remaining_chars < 50:
            break
    if not selected:
        return lead
    # Preserve selected source order for temporal/numeric context.
    order = {piece: pos for _, pos, piece in candidates}
    selected.sort(key=lambda piece: order[piece])
    return lead + " " + " ".join(selected)
