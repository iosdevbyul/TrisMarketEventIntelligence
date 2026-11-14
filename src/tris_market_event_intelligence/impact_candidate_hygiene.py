"""Conservative source and evidence hygiene for *new* impact pilots.

Heuristics only affect candidate selection. No historical event is mutated.
"""
from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Strip only known tracking params; retain article identity such as nttId.
_TRACKING = {"at_medium", "at_campaign", "fbclid", "gclid"}


def article_key(url: str) -> str | None:
    if not url.strip():
        return None
    parsed = urlsplit(url.strip())
    if not parsed.scheme or not parsed.netloc:
        return None
    query = [
        (key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in _TRACKING and not key.lower().startswith("utm_")
    ]
    return urlunsplit((
        parsed.scheme.lower(), parsed.netloc.lower(),
        parsed.path.rstrip("/") or "/", urlencode(sorted(query)), "",
    ))


def evidence_key(evidence: str) -> str | None:
    normalized = re.sub(r"\s+", " ", evidence).strip().casefold()
    # Avoid treating generic short headlines as equivalent claims.
    return normalized if len(normalized) >= 50 else None


def insufficient_evidence(classification: str, evidence: str) -> bool:
    """Exclude obvious title-only statistical releases, not ordinary short news."""
    normalized = re.sub(r"\s+", " ", evidence).strip()
    if not normalized:
        return True
    if classification != "ECONOMIC_INDICATOR":
        return False
    # The data-less BOK 'July/August Producer Price Index (preliminary)' headlines.
    ppi_title = re.fullmatch(
        r"20\d{2}년\s*\d{1,2}월\s*생산자물가지수\s*\(잠정\)", normalized
    )
    return ppi_title is not None
