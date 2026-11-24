"""Resolve EDGAR index documents without treating an index as issuer disclosure."""
from __future__ import annotations

import re
from urllib.parse import urljoin, urlsplit

from tris_market_event_intelligence.sec_primary_content_research_cli import _IndexRows, _text

_INDEX = re.compile(
    r"/Archives/edgar/data/(?P<cik>\d{1,10})/(?P<folder>\d{18})/"
    r"(?P<accession>\d{10}-\d{2}-\d{6})-index\.html?",
    re.IGNORECASE,
)
_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.(?:html?|txt|xml)", re.IGNORECASE)


def index_identity(url: str) -> tuple[str, str] | None:
    parts = urlsplit(url)
    match = _INDEX.fullmatch(parts.path)
    if parts.scheme != "https" or parts.netloc != "www.sec.gov" or not match:
        return None
    if parts.query or parts.fragment:
        return None
    accession = match["accession"]
    if accession.replace("-", "") != match["folder"]:
        return None
    return match["cik"], accession


def documents(raw: bytes, url: str) -> tuple[str, tuple[str, ...]]:
    identity = index_identity(url)
    if identity is None:
        raise ValueError("SEC_INDEX_IDENTITY_INVALID")
    parser = _IndexRows()
    parser.feed(raw.decode("utf-8-sig", errors="replace"))
    prefix = url.rsplit("/", 1)[0] + "/"
    primary: set[str] = set()
    exhibits: set[str] = set()
    for row in parser.rows:
        if len(row) < 4:
            continue
        sequence, _, document, form = row[:4]
        for href in document[1]:
            target = urljoin(url, href)
            if not target.startswith(prefix):
                continue
            name = target.removeprefix(prefix)
            if not _FILE.fullmatch(name) or "-index." in name.lower():
                continue
            if sequence[0] == "1" and form[0] and not form[0].upper().startswith("EX-"):
                primary.add(target)
            elif form[0].upper().startswith(("EX-99", "EX-2", "EX-10")):
                exhibits.add(target)
    if len(primary) != 1:
        raise ValueError("SEC_PRIMARY_DOCUMENT_AMBIGUOUS")
    return next(iter(primary)), tuple(sorted(exhibits))


def visible_text(raw: bytes, url: str) -> str:
    return _text(raw, html=not url.lower().endswith(".txt"))
