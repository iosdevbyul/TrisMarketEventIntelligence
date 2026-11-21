"""Automatic, bounded source-document capture for RSS-discovered BOK news.

RSS discovery is not full-text collection. This module persists official source
bytes, extracted PDF layout and a distinct completeness state. It never changes
the RSS capture or overwrites existing summaries/predictions.
"""
from __future__ import annotations

import hashlib
import io
import sqlite3
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import httpx
from pypdf import PdfReader

from tris_market_event_intelligence.bok_official_pdf_download_cli import (
    MAX_PDF_BYTES, find_pdf_url, trusted,
)
from tris_market_event_intelligence.collector import NewsStore

MAX_PAGE_BYTES = 2_000_000
MIN_BODY_CHARS = 300
MAX_ARTICLES_PER_PASS = 4
RETRY_HOURS = 6


class DocumentError(ValueError):
    """A source document is absent, unsafe or not reliably extractable."""


def _get(client: httpx.Client, url: str, *, limit: int) -> bytes:
    # Only the BOK publisher and its official file host are allowed.
    if not trusted(url):
        raise DocumentError("UNTRUSTED_SOURCE")
    # No implicit cross-site redirects, including intermediate hops.
    current = url
    for _ in range(4):
        response = client.get(current, follow_redirects=False, timeout=30)
        if response.status_code in (301, 302, 303, 307, 308):
            from urllib.parse import urljoin

            destination = urljoin(current, response.headers.get("location", ""))
            if not trusted(destination):
                raise DocumentError("UNTRUSTED_REDIRECT")
            current = destination
            continue
        response.raise_for_status()
        if len(response.content) > limit:
            raise DocumentError("SOURCE_TOO_LARGE")
        return response.content
    raise DocumentError("TOO_MANY_REDIRECTS")


def _pdf_layout(data: bytes) -> tuple[str, tuple[int, ...], int]:
    if len(data) > MAX_PDF_BYTES or not data.startswith(b"%PDF-"):
        raise DocumentError("INVALID_PDF_BYTES")
    try:
        reader = PdfReader(io.BytesIO(data), strict=True)
        if reader.is_encrypted or not 1 <= len(reader.pages) <= 250:
            raise DocumentError("UNSUPPORTED_PDF")
        output: list[str] = []
        sparse: list[int] = []
        for number, page in enumerate(reader.pages, 1):
            layout = page.extract_text(extraction_mode="layout") or ""
            if len(layout.strip()) < 50:
                sparse.append(number)
            output.append(f"[PDF_PAGE_{number}]\n{layout.rstrip()}")
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError("PDF_EXTRACTION_FAILED") from exc
    text = "\n\n".join(output)
    if len(text.strip()) < MIN_BODY_CHARS:
        raise DocumentError("PDF_TEXT_INSUFFICIENT")
    return text, tuple(sparse), len(reader.pages)


def init_document_store(store: NewsStore) -> None:
    store.initialize()
    with store.connect() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS article_source_documents (
                article_id INTEGER PRIMARY KEY REFERENCES articles(id),
                status TEXT NOT NULL CHECK(status IN
                    ('ACQUIRED','NEEDS_REVIEW','FAILED')),
                fetched_at TEXT NOT NULL,
                source_url TEXT NOT NULL,
                attachment_url TEXT,
                pdf_sha256 TEXT,
                page_sha256 TEXT,
                pdf_bytes BLOB,
                extracted_text TEXT,
                sparse_pages TEXT NOT NULL DEFAULT '[]',
                error_code TEXT,
                retry_after TEXT,
                page_count INTEGER NOT NULL DEFAULT 0
            )
        """)


def enrich_pending_bok(
    db_path: Path, *, client: httpx.Client | None = None,
    limit: int = MAX_ARTICLES_PER_PASS,
) -> dict[str, object]:
    """Enrich discovered articles, including old metadata-only rows.

    Bounded to a few requests per pass and a six-hour retry delay. No model,
    price data or synthetic text is used. The original RSS DB rows remain intact.
    """
    if not 1 <= limit <= MAX_ARTICLES_PER_PASS:
        raise ValueError("Limit must be between 1 and 4")
    store = NewsStore(db_path)
    init_document_store(store)
    now = datetime.now(UTC)
    with store.connect() as db:
        rows = db.execute("""
            SELECT a.id, a.url, a.title
            FROM articles a
            LEFT JOIN article_source_documents d ON d.article_id = a.id
            WHERE a.source IN ('bok_all','bok_statistics','bok_monetary','bok_corrections')
              AND (
                  d.article_id IS NULL
                  OR (d.status='FAILED' AND d.retry_after <= ?)
              )
            ORDER BY a.id DESC LIMIT ?
        """, (now.isoformat(), limit)).fetchall()
    owned = client is None
    client = client or httpx.Client(headers={
        "User-Agent": "TrisMarketEventIntelligence/official-public-document-research"
    }, trust_env=False)
    collected = flagged = failed = reused = 0
    attempted = 0
    try:
        for article_id, url, title in rows:
            attempted += 1
            timestamp = datetime.now(UTC).isoformat()
            page_hash = pdf_hash = None
            pdf_data: bytes | None = None
            text = None
            pdf_url = None
            sparse: tuple[int, ...] = ()
            pages = 0
            error = None
            status = "FAILED"
            try:
                # BOK RSS URLs may include http; only official HTTPS is accepted.
                if urlparse(url).hostname not in {"www.bok.or.kr", "bok.or.kr"} or not trusted(url):
                    raise DocumentError("UNTRUSTED_ARTICLE_URL")
                # Reuse identical official article documents across BOK RSS feeds.
                with store.connect() as db:
                    existing = db.execute("""
                        SELECT d.status, d.attachment_url, d.pdf_sha256,
                               d.page_sha256, d.pdf_bytes, d.extracted_text,
                               d.sparse_pages, d.page_count
                        FROM article_source_documents d
                        JOIN articles a ON a.id = d.article_id
                        WHERE a.url=? AND d.status IN ('ACQUIRED','NEEDS_REVIEW')
                        ORDER BY d.article_id DESC LIMIT 1
                    """, (url,)).fetchone()
                if existing is not None:
                    import json

                    status, pdf_url, pdf_hash, page_hash, pdf_data, text, raw_sparse, pages = existing
                    sparse = tuple(json.loads(raw_sparse))
                    reused += 1
                else:
                    html = _get(client, url, limit=MAX_PAGE_BYTES)
                    if b"<html" not in html.lower() and b"<!doctype html" not in html.lower():
                        raise DocumentError("RELEASE_NOT_HTML")
                    page_hash = hashlib.sha256(html).hexdigest()
                    try:
                        pdf_url = find_pdf_url(html.decode("utf-8", errors="replace"), url)
                    except ValueError as exc:
                        raise DocumentError("NO_UNAMBIGUOUS_OFFICIAL_PDF") from exc
                    pdf_data = _get(client, pdf_url, limit=MAX_PDF_BYTES)
                    pdf_hash = hashlib.sha256(pdf_data).hexdigest()
                    text, sparse, pages = _pdf_layout(pdf_data)
                    status = "NEEDS_REVIEW" if sparse else "ACQUIRED"
                if status == "ACQUIRED":
                    collected += 1
                else:
                    flagged += 1
            except (DocumentError, httpx.HTTPError, OSError) as exc:
                error = str(exc) if isinstance(exc, DocumentError) else type(exc).__name__
                failed += 1
            import json

            retry_after = (
                (datetime.now(UTC) + timedelta(hours=RETRY_HOURS)).isoformat()
                if status == "FAILED" else None
            )
            with store.connect() as db:
                db.execute("""
                    INSERT INTO article_source_documents (
                        article_id,status,fetched_at,source_url,attachment_url,
                        pdf_sha256,page_sha256,pdf_bytes,extracted_text,
                        sparse_pages,error_code,retry_after,page_count
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(article_id) DO UPDATE SET
                        status=excluded.status,fetched_at=excluded.fetched_at,
                        attachment_url=excluded.attachment_url,
                        pdf_sha256=excluded.pdf_sha256,page_sha256=excluded.page_sha256,
                        pdf_bytes=excluded.pdf_bytes,extracted_text=excluded.extracted_text,
                        sparse_pages=excluded.sparse_pages,error_code=excluded.error_code,
                        retry_after=excluded.retry_after,page_count=excluded.page_count
                """, (article_id, status, timestamp, url, pdf_url, pdf_hash, page_hash,
                      pdf_data, text, json.dumps(sparse), error, retry_after, pages))
        with store.connect() as db:
            remaining = db.execute("""
                SELECT COUNT(*) FROM articles a
                LEFT JOIN article_source_documents d ON a.id=d.article_id
                WHERE a.source IN ('bok_all','bok_statistics','bok_monetary','bok_corrections')
                  AND (d.article_id IS NULL OR d.status='FAILED')
            """).fetchone()[0]
    finally:
        if owned:
            client.close()
    return {
        "status": "COMPLETE" if remaining == 0 and flagged == 0 and failed == 0
                  else "INCOMPLETE",
        "attempted": attempted, "acquired": collected, "needs_visual_review": flagged,
        "failed": failed, "reused": reused, "pending_or_failed": remaining,
        "article_body_verified": False,
    }
