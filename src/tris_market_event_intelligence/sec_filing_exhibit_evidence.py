"""Read-only official SEC Exhibit 99.x source augmentation for semantic replay.

Primary 8-K covers often contain no earnings figures: those are furnished
in Exhibit 99.1. Fetch only from the issuer's same SEC accession archive
and record exact file URLs, byte hashes, and retrieval failures. Historical
backfills are NOT evidence of first-publication-time availability.
"""
from __future__ import annotations

import hashlib
import os
import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin

import httpx

from tris_market_event_intelligence.sec_issuer_event_acquisition_cli import (
    source_body,
)

MAX_INDEX_BYTES = 1_000_000
MAX_EXHIBIT_BYTES = 2_000_000
SEC_ARCHIVE = re.compile(
    r"https://www\.sec\.gov/Archives/edgar/data/"
    r"(?P<cik>[1-9]\d{0,9})/(?P<accession>\d{18})/"
    r"(?P<document>[A-Za-z0-9][A-Za-z0-9_.-]{0,199})\Z",
)


class _ExhibitRows(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[tuple[str, list[str]]] = []
        self.current_text: list[str] = []
        self.current_links: list[str] = []
        self.in_row = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.in_row = True
            self.current_text = []
            self.current_links = []
        if self.in_row and tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.current_links.append(href)

    def handle_data(self, data: str) -> None:
        if self.in_row:
            self.current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "tr" and self.in_row:
            self.rows.append((
                " ".join(" ".join(self.current_text).split()),
                self.current_links,
            ))
            self.in_row = False


def accession_archive(article: dict[str, Any]) -> tuple[str, str]:
    match = SEC_ARCHIVE.fullmatch(str(article.get("sourceUrl", "")))
    source = str(article.get("source") or "")
    if match is None or source != f'sec_8k_{match.group("cik")}':
        raise ValueError("SEC_EXHIBIT_ARCHIVE_PROVENANCE_MISMATCH")
    base = (
        "https://www.sec.gov/Archives/edgar/data/"
        f'{match.group("cik")}/{match.group("accession")}/'
    )
    digits = match.group("accession")
    accession = f"{digits[:10]}-{digits[10:12]}-{digits[12:]}"
    return base, f"{base}{accession}-index.htm"


def exhibit_urls(index_html: bytes, article: dict[str, Any]) -> dict[str, str]:
    if len(index_html) > MAX_INDEX_BYTES:
        raise ValueError("SEC_EXHIBIT_INDEX_TOO_LARGE")
    base, index_url = accession_archive(article)
    parser = _ExhibitRows()
    parser.feed(index_html.decode("utf-8", errors="replace"))
    links: dict[str, str] = {}
    for text, hrefs in parser.rows:
        for exhibit in ("99.1", "99.2", "10.1"):
            # SEC official filing detail document table includes the
            # 'EX-99.1' type, or a '99.1' exhibit description.
            if not re.search(rf"(?i)\b(?:EX-)?{re.escape(exhibit)}\b", text):
                continue
            for href in hrefs:
                resolved = urljoin(index_url, href)
                if not resolved.startswith(base):
                    continue
                suffix = resolved[len(base):]
                if (
                    not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", suffix)
                    or ".." in suffix
                    or not suffix.lower().endswith((".htm", ".html", ".txt"))
                    or resolved == article["sourceUrl"]
                ):
                    continue
                links.setdefault(exhibit, resolved)
                break
    return links


def enrich_sec_article(
    article: dict[str, Any], *, client: httpx.Client, user_agent: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return an in-memory enriched copy with a per-exhibit audit.

    No article/source journal writes are performed. Only verifiable primary
    SEC archive locations can be used. The LLM receives the source text
    and selected official exhibits, not fetched URLs or price outcomes.
    """
    base, index_url = accession_archive(article)
    if "@" not in user_agent or len(user_agent) > 200:
        raise ValueError("SEC_IDENTIFYING_USER_AGENT_REQUIRED")
    original = str(article.get("summary") or "")
    audit: dict[str, Any] = {
        "status": "NO_VERIFIED_EXHIBIT",
        "filingIndex": index_url,
        "sourceArchive": base,
        "acquiredExhibits": [],
        "missingExhibits": [],
        "firstPublicationPITCertified": False,
    }
    try:
        response = client.get(
            index_url, headers={"User-Agent": user_agent},
        )
        response.raise_for_status()
        links = exhibit_urls(response.content, article)
    except (ValueError, httpx.HTTPError) as exc:
        audit["status"] = "INDEX_UNAVAILABLE"
        audit["errorCode"] = type(exc).__name__
        return article, audit
    parts = [original[:4500]]
    preferred = (
        ("10.1", "99.1")
        if "item 1.01" in str(article.get("title", "")).casefold()
        else ("99.1", "99.2")
    )
    for tag in preferred:
        url = links.get(tag)
        if url is None:
            audit["missingExhibits"].append(tag)
            continue
        try:
            response = client.get(
                url, headers={"User-Agent": user_agent},
            )
            response.raise_for_status()
            if len(response.content) > MAX_EXHIBIT_BYTES:
                raise ValueError("SEC_EXHIBIT_TOO_LARGE")
            excerpt = source_body(response.content)
            if len(excerpt) < 100:
                raise ValueError("SEC_EXHIBIT_NO_SUBSTANTIVE_TEXT")
            # Keep actual numbers and disclosure content, not only the
            # cover page. The text remains clearly marked as retrospective.
            length = 9000 if tag in {"99.1", "10.1"} else 3500
            parts.append(f"SEC EXHIBIT {tag} SOURCE TEXT: {excerpt[:length]}")
            audit["acquiredExhibits"].append({
                "exhibitType": tag, "sourceUrl": url,
                "rawSha256": hashlib.sha256(response.content).hexdigest(),
                "extractedTextSha256": hashlib.sha256(excerpt.encode()).hexdigest(),
                "extractedChars": len(excerpt),
            })
        except (ValueError, httpx.HTTPError) as exc:
            audit["missingExhibits"].append(tag)
            audit.setdefault("errors", {})[tag] = type(exc).__name__
    if audit["acquiredExhibits"]:
        audit["status"] = (
            "EXHIBITS_ACQUIRED"
            if not audit["missingExhibits"] else "PARTIAL_EXHIBITS_ACQUIRED"
        )
        return {**article, "summary": " ".join(parts)}, audit
    return article, audit


def configured_sec_agent() -> str:
    return os.environ.get("SEC_USER_AGENT", "")
