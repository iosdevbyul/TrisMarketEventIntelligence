"""Source-neutral article body acquisition for the existing scheduled RSS collector.

Preserve original discovery rows. A URL is only DISCOVERED until the publisher
body is captured and the content passes a minimum completeness check.
"""
from __future__ import annotations

import hashlib
import html
import io
import json
import re
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from pypdf import PdfReader

from tris_market_event_intelligence.collector import NewsStore

MAX_HTML_BYTES = 2_000_000
MAX_PDF_BYTES = 20_000_000
MAX_ITEMS = 8
MIN_BODY_LENGTH = 250
RETRY_AFTER_HOURS = 6
ALLOWED_HOSTS = {
    "www.bok.or.kr", "bok.or.kr", "file-cdn.bok.or.kr",
    "www.federalreserve.gov", "federalreserve.gov",
    "www.bbc.com", "bbc.com", "www.bbc.co.uk", "bbc.co.uk",
    "feeds.bbci.co.uk", "www.sec.gov", "www.bls.gov",
}


class SourceContentError(ValueError):
    pass


def safe_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return (parsed.scheme == "https" and parsed.hostname in ALLOWED_HOSTS
                and parsed.port in (None, 443) and not parsed.username
                and not parsed.password and len(url) <= 2048)
    except ValueError:
        return False


def fetch_bytes(client: httpx.Client, url: str, limit: int) -> tuple[bytes, str]:
    """Validate every redirect hop; never fetch arbitrary RSS-advertised hosts."""
    for _ in range(4):
        if not safe_url(url):
            raise SourceContentError("UNTRUSTED_PUBLISHER_URL")
        with client.stream("GET", url, follow_redirects=False, timeout=20) as response:
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise SourceContentError("REDIRECT_MISSING")
                url = urljoin(url, location)
                continue
            response.raise_for_status()
            content = bytearray()
            for part in response.iter_bytes():
                content.extend(part)
                if len(content) > limit:
                    raise SourceContentError("SOURCE_TOO_LARGE")
            return bytes(content), response.headers.get("content-type", "").lower()
    raise SourceContentError("REDIRECT_LIMIT")


class ArticleHTML(HTMLParser):
    """Restrict extraction to article/main/recognized story containers."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.level = 0
        self.skip = 0
        self.body: list[str] = []
        self.pdf_links: list[tuple[str, str]] = []
        self.current_link: str | None = None
        self.link_words: list[str] = []
        self._stack: list[tuple[bool, bool]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        skipped = tag in {"script", "style", "noscript", "nav", "footer", "header"}
        if skipped:
            self.skip += 1
        classes = (attributes.get("class") or "").split()
        identifier = attributes.get("id") or ""
        is_body = (tag in {"article", "main"}
                   or identifier in {"dic_area", "articleBodyContents", "article-body",
                                     "article-content", "article_view", "bodyContent"}
                   or any(c in classes for c in {
                       "article-body", "article__body", "story-body",
                       "ssrcss-11r1m41-RichTextComponentWrapper",
                       "news-article", "view-content", "article-content",
                   }))
        if is_body:
            self.level += 1
        elif self.level:
            self.level += 1
        if tag == "a":
            self.current_link = attributes.get("href")
            self.link_words = []
        self._stack.append((skipped, bool(is_body)))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if not self._stack:
            return
        skipped, _ = self._stack.pop()
        if tag == "a" and self.current_link:
            label = " ".join(self.link_words)
            if ".pdf" in (label + self.current_link).lower():
                self.pdf_links.append((self.current_link, label))
            self.current_link = None
            self.link_words = []
        if skipped:
            self.skip = max(0, self.skip - 1)
        if self.level:
            self.level -= 1

    def handle_data(self, data: str) -> None:
        if self.current_link is not None:
            self.link_words.append(data)
        if not self.skip and self.level and data.strip():
            self.body.append(data.strip())


def extract_html(payload: bytes, url: str) -> tuple[str, list[str]]:
    if len(payload) > MAX_HTML_BYTES:
        raise SourceContentError("HTML_TOO_LARGE")
    parser = ArticleHTML()
    parser.feed(payload.decode("utf-8-sig", "replace"))
    body = re.sub(r"\s+", " ", html.unescape(" ".join(parser.body))).strip()
    attachments = list(dict.fromkeys(
        target for href, _ in parser.pdf_links
        if safe_url(target := urljoin(url, href))
    ))
    return body, attachments[:8]


def extract_pdf(data: bytes) -> tuple[str, list[int]]:
    if not data.startswith(b"%PDF-") or len(data) > MAX_PDF_BYTES:
        raise SourceContentError("INVALID_PDF")
    try:
        reader = PdfReader(io.BytesIO(data), strict=True)
        if reader.is_encrypted or not 1 <= len(reader.pages) <= 250:
            raise SourceContentError("UNSUPPORTED_PDF")
        chunks: list[str] = []
        sparse: list[int] = []
        for i, page in enumerate(reader.pages, 1):
            body = page.extract_text(extraction_mode="layout") or ""
            if len(body.strip()) < 50:
                sparse.append(i)
            chunks.append(f"[PDF PAGE {i}]\n{body.rstrip()}")
    except SourceContentError:
        raise
    except Exception as exc:
        raise SourceContentError("PDF_EXTRACTION_FAILED") from exc
    return "\n\n".join(chunks), sparse


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
                content_sha256 TEXT,
                content TEXT,
                format TEXT,
                page_warnings_json TEXT NOT NULL DEFAULT '[]',
                error_code TEXT,
                retry_after TEXT
            )
        """)


def acquire_article(client: httpx.Client, url: str) -> dict:
    payload, mime = fetch_bytes(client, url, MAX_HTML_BYTES)
    digest = hashlib.sha256(payload).hexdigest()
    if payload.startswith(b"%PDF-"):
        body, sparse = extract_pdf(payload)
        return {"status": "REVIEW_REQUIRED" if sparse else "COMPLETE",
                "content": body, "content_url": url, "content_sha256": digest,
                "format": "pdf", "page_warnings_json": json.dumps(sparse)}
    if "html" not in mime and not payload.lstrip().lower().startswith((b"<html", b"<!doctype html")):
        raise SourceContentError("UNSUPPORTED_CONTENT_TYPE")
    body, links = extract_html(payload, url)
    if len(body) >= MIN_BODY_LENGTH:
        return {"status": "COMPLETE", "content": body, "content_url": url,
                "content_sha256": digest, "format": "html",
                "page_warnings_json": "[]"}
    # An HTML-only release may point to its primary PDF body.
    for link in links:
        try:
            pdf, _ = fetch_bytes(client, link, MAX_PDF_BYTES)
            text, sparse = extract_pdf(pdf)
            if len(text.strip()) >= MIN_BODY_LENGTH:
                return {"status": "REVIEW_REQUIRED" if sparse else "COMPLETE",
                        "content": text, "content_url": link,
                        "content_sha256": hashlib.sha256(pdf).hexdigest(),
                        "format": "pdf",
                        "page_warnings_json": json.dumps(sparse)}
        except (SourceContentError, httpx.HTTPError):
            continue
    raise SourceContentError("ARTICLE_BODY_NOT_FOUND")


def collect_full_text(db_path: Path, *, client: httpx.Client | None = None,
                      limit: int = MAX_ITEMS) -> dict[str, object]:
    """Bounded catch-up of newly discovered or previously incomplete RSS news."""
    if not 1 <= limit <= MAX_ITEMS:
        raise ValueError("limit must be between 1 and 8")
    store = NewsStore(db_path)
    initialize(store)
    now = datetime.now(UTC).isoformat()
    with store.connect() as db:
        candidates = db.execute("""
            SELECT a.id, a.url FROM articles a
            LEFT JOIN article_full_text f ON f.article_id=a.id
            WHERE f.article_id IS NULL
               OR (f.status='INCOMPLETE' AND f.retry_after <= ?)
            ORDER BY a.id DESC LIMIT ?
        """, (now, limit)).fetchall()
    own_client = client is None
    client = client or httpx.Client(
        follow_redirects=False, trust_env=False,
        headers={"User-Agent": "TrisMarketEventIntelligence/0.1 (+local research)"},
    )
    counts = {"complete": 0, "review_required": 0, "incomplete": 0}
    try:
        for article_id, url in candidates:
            observed_at = datetime.now(UTC).isoformat()
            try:
                article = acquire_article(client, url)
                error = None
            except (SourceContentError, httpx.HTTPError, OSError, ValueError) as exc:
                error = str(exc) if isinstance(exc, SourceContentError) else type(exc).__name__
                article = {
                    "status": "INCOMPLETE", "content_url": None, "content_sha256": None,
                    "content": None, "format": None, "page_warnings_json": "[]",
                }
            counts[article["status"].lower()] += 1
            retry_after = ((datetime.now(UTC) + timedelta(hours=RETRY_AFTER_HOURS)).isoformat()
                           if article["status"] == "INCOMPLETE" else None)
            with store.connect() as db:
                db.execute("""
                    INSERT INTO article_full_text (
                        article_id,status,observed_at,content_url,content_sha256,
                        content,format,page_warnings_json,error_code,retry_after
                    ) VALUES (?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(article_id) DO UPDATE SET
                        status=excluded.status, observed_at=excluded.observed_at,
                        content_url=excluded.content_url,
                        content_sha256=excluded.content_sha256,
                        content=excluded.content, format=excluded.format,
                        page_warnings_json=excluded.page_warnings_json,
                        error_code=excluded.error_code,retry_after=excluded.retry_after
                """, (article_id, article["status"], observed_at, article["content_url"],
                      article["content_sha256"], article["content"], article["format"],
                      article["page_warnings_json"], error, retry_after))
    finally:
        if own_client:
            client.close()
    with store.connect() as db:
        pending = db.execute("""
            SELECT COUNT(*) FROM articles a
            LEFT JOIN article_full_text f ON f.article_id=a.id
            WHERE f.article_id IS NULL OR f.status!='COMPLETE'
        """).fetchone()[0]
    return {"status": "COMPLETE" if pending == 0 else "INCOMPLETE",
            "attempted": len(candidates), **counts, "pending_or_unverified": pending,
            "rss_discovery_is_full_collection": False}
