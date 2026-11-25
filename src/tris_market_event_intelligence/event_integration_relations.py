"""Evidence-bearing cross-source retrieval, never automatic event confirmation."""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import UTC, datetime, timedelta

# Series identity, not a claim that the stored value was the original release value.
INDICATORS = {
    "US_CPI": (
        r"\b(?:consumer price index|CPI)\b|소비자물가",
        {"bls_cpi", "bls_latest"},
        {("FRED_CSV", "CPIAUCSL"), ("FRED", "CPIAUCSL"), ("BLS", "CUSR0000SA0")},
    ),
    "US_PPI": (
        r"\b(?:producer price index|PPI)\b|생산자물가",
        {"bls_ppi", "bls_latest"},
        {("FRED_CSV", "PPIACO"), ("FRED", "PPIACO")},
    ),
    "US_PAYROLL": (
        r"\b(?:nonfarm payrolls?|employment situation)\b|비농업\s*고용",
        {"bls_employment", "bls_latest"},
        {("FRED_CSV", "PAYEMS"), ("FRED", "PAYEMS"), ("BLS", "CES0000000001")},
    ),
    "US_GDP": (
        r"\b(?:gross domestic product|GDP)\b|국내총생산",
        {"bea_news"},
        {("FRED_CSV", "GDPC1"), ("FRED", "GDPC1")},
    ),
    "US_PERMITS": (
        r"\b(?:building permits?|new residential construction)\b|건축허가",
        {"US_census"},
        {("FRED_CSV", "PERMIT"), ("FRED", "PERMIT")},
    ),
}
US = re.compile(r"(?i)\b(?:U\.S\.|US|United States|American|BLS|BEA)\b|미국")
MONTHS = {
    name: i for i, name in enumerate(
        "january february march april may june july august september october november december".split(), 1
    )
}
MONTH = re.compile(
    r"(?i)\b(" + "|".join(MONTHS) + r")\s+(\d{4})\b|(\d{4})년\s*(1[0-2]|0?[1-9])월"
)
QUARTER = re.compile(
    r"(?i)\b(?:Q([1-4])|([1-4])(?:st|nd|rd|th)\s+quarter|"
    r"(first|second|third|fourth)\s+quarter)[,\s]+(\d{4})\b"
    r"|(\d{4})년\s*([1-4])분기"
)
CORRECTION = re.compile(r"(?i)\b(?:corrects?|correction|retracts?|denies|denial)\b|정정|부인|반박")
FOLLOW = re.compile(r"(?i)\b(?:follow-up|update|revised|revision|subsequently)\b|후속|수정|추가 발표")
FORECAST = re.compile(r"(?i)\b(?:forecast|preview|expected|anticipates?)\b|전망|예상|발표 예정")


def periods(text, quarterly=False):
    result = set()
    if quarterly:
        for m in QUARTER.finditer(text):
            first, numeric, word, year, kr_year, kr_q = m.groups()
            q = first or numeric or kr_q
            if word:
                q = str(("first", "second", "third", "fourth").index(word.lower()) + 1)
            result.add(f"{year or kr_year}-{(int(q) - 1) * 3 + 1:02}-01")
    else:
        for m in MONTH.finditer(text):
            name, year, kr_year, month = m.groups()
            result.add(f"{year or kr_year}-{MONTHS[name.lower()] if name else int(month):02}-01")
    return result


def features(kind, source, title, body):
    """Exact text spans; period inferred only from unambiguous explicit title."""
    indicators = []
    for key, (pattern, sources, _) in INDICATORS.items():
        if kind == "ECONOMIC_RELEASE" and source not in sources:
            continue
        if kind == "NEWS" and not US.search(title):
            continue
        if kind not in {"ECONOMIC_RELEASE", "NEWS"}:
            continue
        match = re.search(pattern, title, re.IGNORECASE)
        if not match:
            continue
        dates = periods(title, quarterly=key == "US_GDP")
        if len(dates) == 1:
            indicators.append({"indicator": key, "period": next(iter(dates)),
                               "titleSpan": match.group(), "periodScope": "TITLE_EXPLICIT"})
    accessions = sorted(set(re.findall(r"\b\d{10}-\d{2}-\d{6}\b", body)))[:20]
    # Exact quoted phrase overlap: strong lexical evidence but not semantic verification.
    phrases = set()
    for match in re.finditer(r'["“]([^"”\n]{35,250})["”]', body[:50000]):
        phrase = " ".join(match[1].split()).casefold()
        if len(phrase.split()) >= 7:
            phrases.add(phrase)
    return {"indicators": indicators, "accessionsMentioned": accessions,
            "quotedPhrases": sorted(phrases)[:30],
            "correctionCue": bool(CORRECTION.search(title)),
            "followUpCue": bool(FOLLOW.search(title)),
            "forecastCue": bool(FORECAST.search(title))}


def stamp(value):
    try:
        result = datetime.fromisoformat(value or "")
        return result.astimezone(UTC) if result.utcoffset() is not None else None
    except (ValueError, TypeError):
        return None


def declared(item):
    return stamp(item.get("publisherDeclaredAt") or item.get("secAcceptanceAt"))


def close_dates(a, b):
    x, y = declared(a), declared(b)
    return bool(x and y and a.get("declaredClockConsistent") and b.get("declaredClockConsistent")
                and abs(x - y) <= timedelta(days=7))


def cross_source_edges(events):
    """Explicit accession, shared quote, indicator/period and series/period edges."""
    filings = defaultdict(list)
    observations = defaultdict(list)
    releases = defaultdict(list)
    quotes = defaultdict(list)
    articles = []
    for item in events:
        kind = item["kind"]
        if kind == "ECONOMIC_OBSERVATION":
            if item["quality"] == "NUMERIC_CONTEXT_ONLY":
                observations[(item["source"], item["series"], item["period"])].append(item)
            continue
        if not item.get("captureVerified"):
            continue
        if item.get("secIdentity"):
            filings[item["secIdentity"][1]].append(item)
        f = item.get("integrationFeatures", {})
        if kind == "ECONOMIC_RELEASE":
            for i in f.get("indicators", []):
                releases[(i["indicator"], i["period"])].append((item, i))
        if item["quality"] not in {"TEXT_CANDIDATE", "ATTACHMENT_TEXT_CANDIDATE"}:
            continue
        articles.append(item)
        for q in f.get("quotedPhrases", []):
            quotes[q].append(item)
    for item in articles:
        if item["kind"] != "NEWS":
            continue
        f = item.get("integrationFeatures", {})
        if f.get("forecastCue") or item.get("claimKind") in {"FORECAST", "RUMOR"}:
            continue
        for accession in f.get("accessionsMentioned", []):
            for filing in filings[accession][:10]:
                yield filing, item, "SEC_ACCESSION_MENTION_CANDIDATE", {"accession": accession}
        for i in f.get("indicators", []):
            for release, other in releases[(i["indicator"], i["period"])][:20]:
                if close_dates(release, item):
                    yield release, item, "INDICATOR_REPORT_CANDIDATE", {"news": i, "release": other}
    for pairs in releases.values():
        for release, i in pairs:
            if release.get("integrationFeatures", {}).get("forecastCue"):
                continue
            for source, series in sorted(INDICATORS[i["indicator"]][2]):
                for obs in observations[(source, series, i["period"])][:20]:
                    yield release, obs, "RELEASE_SERIES_PERIOD_CONTEXT", {
                        "release": i, "series": series, "source": source,
                        "originalReleaseValueVerified": False,
                    }
    for quote, group in quotes.items():
        official = [item for item in group if item["kind"] in {"SEC", "ECONOMIC_RELEASE"}]
        news = [item for item in group if item["kind"] == "NEWS"]
        for a in official[:20]:
            for b in news[:20]:
                if close_dates(a, b):
                    yield a, b, "SHARED_QUOTATION_CANDIDATE", {"normalizedQuote": quote}
