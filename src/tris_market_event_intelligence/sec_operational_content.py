"""Resolve EDGAR index documents without treating an index as issuer disclosure."""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from urllib.parse import parse_qs, urljoin, urlsplit

from tris_market_event_intelligence.sec_primary_content_research_cli import _IndexRows

_INDEX = re.compile(
    r"/Archives/edgar/data/(?P<cik>\d{1,10})/(?P<folder>\d{18})/"
    r"(?P<accession>\d{10}-\d{2}-\d{6})-index\.html?",
    re.IGNORECASE,
)
_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.(?:html?|txt|xml|pdf|xsd|json|csv|jpg|jpeg|png|gif|xlsx|zip)", re.IGNORECASE)


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
    source_match = re.fullmatch(r"sec_(?:8k|filing)_(\d{1,10})", source)
    if source_match is None:
        raise ValueError("SEC_SOURCE_ID_INVALID")
    if int(cik) != int(source_match[1]):
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


def document_name(name: str) -> str | None:
    # Ownership forms link to an XSL-rendered view, not an archive subdirectory.
    name = re.sub(r"^xsl[A-Za-z0-9_-]+/", "", name)
    return name if _FILE.fullmatch(name) else None


def primary_from_submissions(raw: bytes, accession: str) -> str | None:
    packet = json.loads(raw)
    columns = packet.get("filings", {}).get("recent", packet)
    numbers = columns.get("accessionNumber", [])
    names = columns.get("primaryDocument", [])
    if len(numbers) != len(names):
        return None
    matches = {name for number, name in zip(numbers, names, strict=True)
               if number == accession and isinstance(name, str) and name}
    return document_name(next(iter(matches))) if len(matches) == 1 else None


def documents(raw: bytes, url: str, *,
              primary_name: str | None = None) -> tuple[str, tuple[str, ...]]:
    identity = index_identity(url)
    if identity is None:
        raise ValueError("SEC_INDEX_IDENTITY_INVALID")
    parser = _IndexRows()
    parser.feed(raw.decode("utf-8-sig", errors="replace"))
    prefix = url.rsplit("/", 1)[0] + "/"
    primary: set[str] = set()
    exhibits: set[str] = set()
    authoritative: set[str] = set()
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
            name = document_name(target.removeprefix(prefix))
            if name is None or "-index." in name.lower():
                continue
            target = prefix + name
            if primary_name == name and form[0] and not form[0].upper().startswith("EX-"):
                authoritative.add(target)
            if sequence[0] == "1" and form[0] and not form[0].upper().startswith("EX-"):
                primary.add(target)
            elif re.match(r"EX-(?:99|2|10)(?:\.|$)", form[0].upper()):
                exhibits.add(target)
    if primary_name is not None:
        if len(authoritative) != 1:
            raise ValueError("SEC_PRIMARY_DOCUMENT_METADATA_MISMATCH")
        primary = authoritative
    if len(primary) != 1:
        raise ValueError("SEC_PRIMARY_DOCUMENT_AMBIGUOUS")
    return next(iter(primary)), tuple(sorted(exhibits - primary))


def visible_text(raw: bytes, url: str) -> str:
    if not raw or len(raw) > 8_000_000:
        raise ValueError("SEC_DOCUMENT_SIZE_INVALID")
    if url.lower().endswith(".xml"):
        if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
            raise ValueError("SEC_XML_DTD_UNSUPPORTED")
        try:
            root = ET.fromstring(raw)
        except ET.ParseError as exc:
            raise ValueError("SEC_XML_INVALID") from exc
        # Keep hierarchical fields and literal values; preserve transaction boundaries.
        lines: list[str] = []

        def visit(element: ET.Element, parent: str = "") -> None:
            path = parent + "/" + element.tag.rsplit("}", 1)[-1]
            if element.text and element.text.strip():
                lines.append(f"{path}: {element.text.strip()}")
            for child in element:
                visit(child, path)

        visit(root)
        return "\n".join(lines)
    decoded = raw.decode("utf-8-sig", errors="replace")
    if url.lower().endswith(".txt") and not re.search(r"<(?:html|document|text)\b", decoded, re.IGNORECASE):
        return " ".join(decoded.split())

    class Visible(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self.parts: list[str] = []
            self.hidden: list[str] = []

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            if tag in {"script", "style", "ix:hidden", "ix:header", "head"}:
                self.hidden.append(tag)

        def handle_endtag(self, tag: str) -> None:
            if tag in self.hidden:
                while self.hidden.pop() != tag:
                    pass

        def handle_data(self, data: str) -> None:
            if not self.hidden:
                self.parts.append(data)

    parser = Visible()
    parser.feed(decoded)
    parser.close()
    return " ".join(" ".join(parser.parts).split())
