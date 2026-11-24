"""Resolve EDGAR index documents without treating an index as issuer disclosure."""
from __future__ import annotations

import re
from urllib.parse import parse_qs, urljoin, urlsplit

from tris_market_event_intelligence.sec_primary_content_research_cli import _IndexRows, _text

_INDEX = re.compile(
    r"/Archives/edgar/data/(?P<cik>\d{1,10})/(?P<folder>\d{18})/"
    r"(?P<accession>\d{10}-\d{2}-\d{6})-index\.html?",
    re.IGNORECASE,
)
_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.(?:html?|txt|xml)", re.IGNORECASE)


def canonical_index(url: str, source: str) -> str:
    """Resolve direct, index and inline-viewer links within the issuer archive."""
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.netloc != "www.sec.gov":
        raise ValueError("SEC_SOURCE_URL_INVALID")
    if parts.path in ("/ix", "/ixviewer/doc/action"):
        docs = parse_qs(parts.query).get("doc", [])
        if len(docs) != 1 or not docs[0].startswith("/Archives/edgar/data/"):
            raise ValueError("SEC_VIEWER_DOCUMENT_INVALID")
        parts = urlsplit("https://www.sec.gov" + docs[0])
    match = re.fullmatch(
        r"/Archives/edgar/data/(\d{1,10})/(\d{18})/([A-Za-z0-9_.-]+)",
        parts.path, re.IGNORECASE,
    )
    if not match or parts.query or parts.fragment:
        raise ValueError("SEC_ARCHIVE_PATH_INVALID")
    cik, folder, name = match.groups()
    if not source.startswith("sec_8k_") or not source.removeprefix("sec_8k_").isdigit():
        raise ValueError("SEC_SOURCE_ID_INVALID")
    if int(cik) != int(source.removeprefix("sec_8k_")):
        raise ValueError("SEC_SOURCE_CIK_MISMATCH")
    if not _FILE.fullmatch(name):
        raise ValueError("SEC_DOCUMENT_NAME_INVALID")
    accession = f"{folder[:10]}-{folder[10:12]}-{folder[12:]}"
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/{accession}-index.htm"


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
            parsed = urlsplit(target)
            if parsed.netloc == "www.sec.gov" and parsed.path in ("/ix", "/ixviewer/doc/action"):
                docs = parse_qs(parsed.query).get("doc", [])
                if len(docs) == 1 and docs[0].startswith("/Archives/edgar/data/"):
                    target = "https://www.sec.gov" + docs[0]
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
