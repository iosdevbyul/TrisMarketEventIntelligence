"""Capture article bodies for every supported operational RSS source.

RSS titles and summaries are discovery evidence, not completed article collection.
Original RSS rows remain immutable; publisher text and retrieval failures are
durably recorded in a separate table.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx
from pypdf import PdfReader

from tris_market_event_intelligence.collector import NewsStore

# Only publishers configured for operational RSS. Never fetch arbitrary hosts
# advertised inside RSS, attachments or redirects.
PUBLISHER_HOSTS = frozenset({
    "www.federalreserve.gov", "federalreserve.gov",
    "www.bok.or.kr", "bok.or.kr", "file-cdn.bok.or.kr",
    "www.bbc.com", "bbc.com", "www.bbc.co.uk", "bbc.co.uk",
    "feeds.bbci.co.uk", "www.sec.gov", "www.bls.gov",
})
MAX_HTML_BYTES = 2_000_000
MAX_PDF_BYTES = 20_000_000
MAX_PAGES = 250
MAX_PER_CYCLE = 12
MIN_TEXT_LENGTH = 250
RETRY_HOURS = 6
USER_AGENT = "TrisMarketEventIntelligence/0.1 (+local research)"


class ContentError(ValueError):
    pass


def allowed(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        return (parsed.scheme == "https" and parsed.hostname in PUBLISHER_HOSTS
                and parsed.port in (None, 443) and parsed.username is None
                and parsed.password is None and len(url) <= 2048)
    except ValueError:
        return False


def fetch(client: httpx.Client, url: str, limit: int) -> tuple[bytes, str, str]:
    for _ in range(4):
        if not allowed(url):
            raise ContentError("UNTRUSTED_URL")
        with client.stream("GET", url, follow_redirects=False, timeout=20) as response:
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise ContentError("MISSING_REDIRECT_LOCATION")
                url = urljoin(url, location)
                continue
            response.raise_for_status()
            payload = bytearray()
            for chunk in response.iter_bytes():
                payload.extend(chunk)
                if len(payload) > limit:
                    raise ContentError("SOURCE_TOO_LARGE")
            return bytes(payload), response.headers.get("content-type", "").lower(), url
    raise ContentError("REDIRECT_LIMIT")


class ArticleParser(HTMLParser):
    """Extract only marked article-body regions and collect PDF links."""

    BODY_IDS = frozenset({
        "article-body", "articleBodyContents", "article-content", "dic_area",
        "bodyContent", "article_view", "main-content", "harmonyContainer",
    })
    BODY_CLASSES = frozenset({
        "article-body", "article__body", "article-content", "story-body",
        "news-article", "article_view", "articleCont",
        "ssrcss-11r1m41-RichTextComponentWrapper",
    })

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.skipped = 0
        self.body: list[str] = []
        self.links: list[tuple[str, str]] = []
        self.anchor: str | None = None
        self.anchor_label: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag in {"script", "style", "noscript", "nav", "footer"}:
            self.skipped += 1
        if tag == "a":
            self.anchor = (
                a.get("href") or a.get("data-url")
                or a.get("data-href") or a.get("data-download-url")
            )
            self.anchor_label = []
        if self.depth:
            self.depth += 1
        elif (not self.skipped and (
            tag == "article"
            or (a.get("id") or "") in self.BODY_IDS
            or bool(set((a.get("class") or "").split()) & self.BODY_CLASSES)
        )):
            self.depth = 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.anchor:
            label = " ".join(self.anchor_label)
            if (".pdf" in (label + self.anchor).lower()
                    or re.search(r"\bpdf\b", label, re.IGNORECASE)):
                self.links.append((self.anchor, label))
            self.anchor = None
            self.anchor_label = []
        if self.depth:
            self.depth -= 1
        if tag in {"script", "style", "noscript", "nav", "footer"} and self.skipped:
            self.skipped -= 1

    def handle_data(self, data: str) -> None:
        if self.anchor is not None:
            self.anchor_label.append(data)
        if self.depth and not self.skipped and data.strip():
            self.body.append(data.strip())


def extract_html(payload: bytes, page_url: str) -> tuple[str, tuple[str, ...]]:
    if len(payload) > MAX_HTML_BYTES:
        raise ContentError("HTML_TOO_LARGE")
    parser = ArticleParser()
    parser.feed(payload.decode("utf-8-sig", errors="replace"))
    body = re.sub(r"\s+", " ", " ".join(parser.body)).strip()
    links = tuple(dict.fromkeys(
        target for href, _ in parser.links
        if allowed(target := urljoin(page_url, href))
    ))
    return body, links[:8]


def extract_pdf(payload: bytes) -> tuple[str, tuple[int, ...]]:
    if not payload.startswith(b"%PDF-") or len(payload) > MAX_PDF_BYTES:
        raise ContentError("INVALID_PDF")
    try:
        reader = PdfReader(io.BytesIO(payload), strict=True)
        if reader.is_encrypted or not 1 <= len(reader.pages) <= MAX_PAGES:
            raise ContentError("UNSUPPORTED_PDF")
        parts: list[str] = []
        sparse: list[int] = []
        for number, page in enumerate(reader.pages, 1):
            page_text = page.extract_text(extraction_mode="layout") or ""
            if len(page_text.strip()) < 50:
                sparse.append(number)
            parts.append(f"[PDF PAGE {number}]\n{page_text.rstrip()}")
    except ContentError:
        raise
    except Exception as exc:
        raise ContentError("PDF_EXTRACTION_FAILED") from exc
    return "\n\n".join(parts), tuple(sparse)


def capture(client: httpx.Client, url: str) -> dict[str, object]:
    payload, media_type, final_url = fetch(client, url, MAX_PDF_BYTES)
    if payload.startswith(b"%PDF-"):
        body, sparse = extract_pdf(payload)
        source_data = payload
        source_url = final_url
        kind = "PDF"
    elif ("html" in media_type or payload.lstrip().lower().startswith(
        (b"<html", b"<!doctype html")
    )):
        body, pdf_links = extract_html(payload, final_url)
        source_data, source_url, kind, sparse = payload, final_url, "HTML", ()
        if len(body) < MIN_TEXT_LENGTH:
            for pdf_url in pdf_links:
                try:
                    candidate, _, resolved = fetch(client, pdf_url, MAX_PDF_BYTES)
                    extracted, candidate_sparse = extract_pdf(candidate)
                    if len(extracted.strip()) >= MIN_TEXT_LENGTH:
                        body, sparse = extracted, candidate_sparse
                        source_data, source_url, kind = candidate, resolved, "PDF"
                        break
                except (httpx.HTTPError, ContentError):
                    continue
    else:
        raise ContentError("UNSUPPORTED_MEDIA_TYPE")
    if len(body.strip()) < MIN_TEXT_LENGTH:
        raise ContentError("INSUFFICIENT_ARTICLE_TEXT")
    return {
        "status": "REVIEW_REQUIRED" if sparse else "COMPLETE",
        "body": body,
        "body_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "raw_sha256": hashlib.sha256(source_data).hexdigest(),
        "raw_payload": source_data,
        "source_url": source_url,
        "format": kind,
        "sparse_pages": json.dumps(sparse),
    }


def initialize(store: NewsStore) -> None:
    store.initialize()
    with store.connect() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS article_full_text (
                article_id INTEGER PRIMARY KEY REFERENCES articles(id),
                status TEXT NOT NULL CHECK(status IN
                    ('COMPLETE','REVIEW_REQUIRED','INCOMPLETE')),
                observed_at TEXT NOT NULL,
                content_url TEXT,
                body_sha256 TEXT,
                raw_sha256 TEXT,
                raw_payload BLOB,
                body TEXT,
                format TEXT,
                sparse_pages TEXT NOT NULL DEFAULT '[]',
                error_code TEXT,
                retry_after TEXT
            )
        """)


def collect_full_text(
    db_path: Path, *, client: httpx.Client | None = None,
    limit: int = MAX_PER_CYCLE,
) -> dict[str, object]:
    """Catch up on previously discovered RSS rows without mutating them."""
    if not 1 <= limit <= MAX_PER_CYCLE:
        raise ValueError("limit must be 1..12")
    store = NewsStore(db_path)
    initialize(store)
    now = datetime.now(UTC).isoformat()
    with store.connect() as db:
        rows = db.execute("""
            SELECT a.id, a.url
            FROM articles a
            LEFT JOIN article_full_text f ON f.article_id=a.id
            WHERE (f.article_id IS NULL OR
                   (f.status='INCOMPLETE' AND f.retry_after <= ?))
              AND a.source IN (
                'fed_monetary','fed_all','bok_all','bok_statistics',
                'bok_corrections','bbc_business','bbc_technology','bbc_world'
              )
            ORDER BY a.id DESC LIMIT ?
        """, (now, limit)).fetchall()
    owned = client is None
    client = client or httpx.Client(
        trust_env=False, follow_redirects=False,
        headers={"User-Agent": USER_AGENT},
    )
    counts = {"complete": 0, "review_required": 0, "incomplete": 0, "reused": 0}
    try:
        for article_id, url in rows:
            observed = datetime.now(UTC).isoformat()
            error: str | None = None
            try:
                with store.connect() as db:
                    existing = db.execute("""
                        SELECT status,content_url,body_sha256,raw_sha256,
                               raw_payload,body,format,sparse_pages
                        FROM article_full_text
                        WHERE status IN ('COMPLETE','REVIEW_REQUIRED')
                          AND article_id IN (SELECT id FROM articles WHERE url=?)
                        LIMIT 1
                    """, (url,)).fetchone()
                if existing:
                    row = dict(zip(("status","source_url","body_sha256",
                                    "raw_sha256","raw_payload","body",
                                    "format","sparse_pages"), existing))
                    counts["reused"] += 1
                else:
                    row = capture(client, url)
            except (ContentError, httpx.HTTPError, OSError) as exc:
                error = str(exc) if isinstance(exc, ContentError) else type(exc).__name__
                row = {
                    "status": "INCOMPLETE", "source_url": None,
                    "body_sha256": None, "raw_sha256": None,
                    "raw_payload": None, "body": None, "format": None,
                    "sparse_pages": "[]",
                }
            counts[str(row["status"]).lower()] += 1
            retry = ((datetime.now(UTC) + timedelta(hours=RETRY_HOURS)).isoformat()
                     if row["status"] == "INCOMPLETE" else None)
            with store.connect() as db:
                db.execute("""
                    INSERT INTO article_full_text (
                        article_id,status,observed_at,content_url,
                        body_sha256,raw_sha256,raw_payload,body,format,
                        sparse_pages,error_code,retry_after
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(article_id) DO UPDATE SET
                        status=excluded.status,observed_at=excluded.observed_at,
                        content_url=excluded.content_url,
                        body_sha256=excluded.body_sha256,
                        raw_sha256=excluded.raw_sha256,
                        raw_payload=excluded.raw_payload,body=excluded.body,
                        format=excluded.format,sparse_pages=excluded.sparse_pages,
                        error_code=excluded.error_code,retry_after=excluded.retry_after
                """, (article_id, row["status"], observed, row["source_url"],
                      row["body_sha256"], row["raw_sha256"], row["raw_payload"],
                      row["body"], row["format"], row["sparse_pages"], error, retry))
    finally:
        if owned:
            client.close()
    with store.connect() as db:
        pending = db.execute("""
            SELECT COUNT(*) FROM articles a
            LEFT JOIN article_full_text f ON f.article_id=a.id
            WHERE a.source IN (
                'fed_monetary','fed_all','bok_all','bok_statistics',
                'bok_corrections','bbc_business','bbc_technology','bbc_world')
              AND (f.article_id IS NULL OR f.status!='COMPLETE')
        """).fetchone()[0]
    return {
        "status": "COMPLETE" if pending == 0 else "INCOMPLETE",
        "attempted": len(rows), **counts, "pending_or_unverified": pending,
    }
