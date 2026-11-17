"""As-of evidence eligibility for historical qualitative news analysis.

An exact publisher quote can still contain post-cutoff price information.
Only a quote which passes the no-observed-market-outcome screen may be
submitted to the model as source support. The original source is untouched.
This screen is deliberately conservative; it does NOT establish the
publication time of the remaining statements.
"""
from __future__ import annotations

import re

# Observed numeric returns and realized directional market moves.
_NUMERIC_MOVE = re.compile(
    r"\b(?:fell|fallen|dropped|declined|slumped|plunged|"
    r"rose|rallied|gained|lost|down|up|drop|decline|"
    r"gain|loss|fall|rise)\b.{0,48}?\b\d+(?:\.\d+)?\s*%|"
    r"\b\d+(?:\.\d+)?\s*%\s*(?:drop|fall|decline|"
    r"gain|rise|rally|slump|loss)\b",
    re.IGNORECASE,
)
_OBSERVED_MOVE = re.compile(
    r"\b(?:shares?|stocks?|equities|market|indices|index|nasdaq|s&p\s*500)\b"
    r".{0,100}?\b(?:dropped|fell|declined|slumped|plunged|"
    r"rallied|rose|gained|lost|sold\s+off|downward)\b|"
    r"\b(?:shares?|stocks?|equities|market|nasdaq|s&p\s*500)\b"
    r".{0,80}?\b(?:is|was|were)\s+(?:down|up)\s+\d",
    re.IGNORECASE,
)
_AS_OF_PRICE = re.compile(
    r"\b(?:share|stock|index|market)\s+price\b.{0,60}"
    r"\b(?:closed|settled|traded|ended)\b",
    re.IGNORECASE,
)


def is_observed_market_result(text: str) -> bool:
    """Quarantine purported price/index outcomes even when quoted verbatim."""
    return bool(
        _NUMERIC_MOVE.search(text)
        or _OBSERVED_MOVE.search(text)
        or _AS_OF_PRICE.search(text)
    )


def filter_evidence_quotes(anchors: list[dict]) -> tuple[list[dict], list[dict]]:
    eligible: list[dict] = []
    quarantined: list[dict] = []
    for anchor in anchors:
        if is_observed_market_result(anchor["text"]):
            quarantined.append({
                "quoteId": anchor["quoteId"],
                "articleId": anchor["articleId"],
                "reason": "OBSERVED_MARKET_OUTCOME_WITHHELD",
            })
        else:
            eligible.append(anchor)
    return eligible, quarantined


def redact_market_result_sentences(text: str) -> str:
    """Scrub model-facing text in memory; never modify the saved source."""
    # Sentence delimiters are kept by splitting only on whitespace.
    sentences = re.split(r"(?<=[.!?])\s+|[\r\n]+", text)
    return " ".join(
        "[MARKET_OUTCOME_WITHHELD_FROM_AS_OF_ANALYSIS]"
        if is_observed_market_result(segment) else segment
        for segment in sentences if segment
    )


# Source evidence naming one issuer is not in itself evidence that
# comparable businesses face the same economic exposure.
_SECTOR_WIDE_LANGUAGE = re.compile(
    r"\b(?:industry(?:-wide)?|sector(?:-wide)?|"
    r"across\s+(?:chip|technology|tech|semiconductor)\s+companies|"
    r"other\s+(?:chipmakers|chip\s+firms|technology\s+companies|"
    r"semiconductor\s+companies)|chipmakers|competitors|suppliers|"
    r"supply\s+chain|multiple\s+(?:firms|companies))\b",
    re.IGNORECASE,
)


def has_sector_wide_source_evidence(excerpts: list[str]) -> bool:
    """Conservative sector-scope cue; not proof of financial causality."""
    return any(_SECTOR_WIDE_LANGUAGE.search(text) for text in excerpts)
