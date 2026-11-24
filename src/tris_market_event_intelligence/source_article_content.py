"""Capture article bodies for every supported operational RSS source.

RSS titles and summaries are discovery evidence, not completed article collection.
Original RSS rows remain immutable; publisher text and retrieval failures are
durably recorded in a separate table.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import time
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx
from pypdf import PdfReader

from tris_market_event_intelligence.collector import NewsStore
from tris_market_event_intelligence.news_content_quality import classify, media_page
from tris_market_event_intelligence.sec_operational_content import (
    canonical_index,
    documents,
    index_identity,
    visible_text,
)

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
    """Capture semantic article regions without confusing nested closing tags.

    Multiple candidates are retained because sites often put the story inside
    a generic <main> element or a publisher-specific content container.
    """

    BODY_IDS = frozenset({
        "article-body", "articleBodyContents", "article-content", "dic_area",
        "bodyContent", "article_view", "main-content", "harmonyContainer",
        "content", "article", "story", "main", "news-article",
    })
    BODY_CLASSES = frozenset({
        "article-body", "article__body", "article-content", "story-body",
        "news-article", "article_view", "articleCont", "story",
        "article__content", "articleBody", "article", "body",
        "ssrcss-11r1m41-RichTextComponentWrapper",
    })
    VOID_TAGS = frozenset({
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    })
    IGNORED_TAGS = frozenset({
        "script", "style", "noscript", "nav", "footer", "aside",
    })

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.frames: list[tuple[str, int | None, bool]] = []
        self.regions: list[tuple[int, list[str]]] = []
        self.links: list[tuple[str, str]] = []
        self.anchors: list[tuple[str | None, list[str]]] = []
        self.ignored = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "a":
            href = (attributes.get("href") or attributes.get("data-url")
                    or attributes.get("data-href")
                    or attributes.get("data-download-url"))
            if not href:
                matches = re.findall(
                    r"""['"]((?:https://|/)[^'"]+)['"]""",
                    attributes.get("onclick") or "",
                )
                href = matches[0] if matches else None
            self.anchors.append((href, []))
        excluded = tag in self.IGNORED_TAGS
        if excluded:
            self.ignored += 1
        region: int | None = None
        if not self.ignored:
            classes = set((attributes.get("class") or "").split())
            identifier = attributes.get("id") or ""
            role = attributes.get("role") or ""
            if tag == "article":
                priority = 4
            elif identifier in self.BODY_IDS or bool(classes & self.BODY_CLASSES):
                priority = 3
            elif tag == "main" or role == "main":
                priority = 2
            else:
                priority = 0
            if priority:
                region = len(self.regions)
                self.regions.append((priority, []))
        if tag not in self.VOID_TAGS:
            self.frames.append((tag, region, excluded))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        # Close through the matching element (HTML often omits optional closes).
        match = next(
            (idx for idx in range(len(self.frames) - 1, -1, -1)
             if self.frames[idx][0] == tag),
            None,
        )
        if match is not None:
            removed = self.frames[match:]
            del self.frames[match:]
            self.ignored = max(
                0, self.ignored - sum(int(frame[2]) for frame in removed)
            )
        if tag == "a" and self.anchors:
            href, words = self.anchors.pop()
            label = " ".join(words)
            if href and (
                ".pdf" in (label + href).lower()
                or re.search(r"\bpdf\b", label, re.IGNORECASE)
            ):
                self.links.append((href, label))

    def handle_data(self, data: str) -> None:
        if self.anchors:
            self.anchors[-1][1].append(data)
        if self.ignored or not data.strip():
            return
        active = {region for _, region, _ in self.frames if region is not None}
        for region in active:
            self.regions[region][1].append(data.strip())


def extract_html(payload: bytes, page_url: str) -> tuple[str, tuple[str, ...]]:
    if len(payload) > MAX_HTML_BYTES:
        raise ContentError("HTML_TOO_LARGE")
    parser = ArticleParser()
    parser.feed(payload.decode("utf-8-sig", errors="replace"))
    candidates = [
        (priority, re.sub(r"\s+", " ", " ".join(parts)).strip())
        for priority, parts in parser.regions
    ]
    candidates.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    # Prefer a specific article region, but allow the general main container
    # when the publisher does not mark its body explicitly.
    body = next(
        (text for _, text in candidates if classify(text) == "TEXT_CANDIDATE"),
        candidates[0][1] if candidates else "",
    )
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
            if not page_text.strip():
                sparse.append(number)
            parts.append(f"[PDF PAGE {number}]\n{page_text.rstrip()}")
    except ContentError:
        raise
    except Exception as exc:
        raise ContentError("PDF_EXTRACTION_FAILED") from exc
    return "\n\n".join(parts), tuple(sparse)


def capture_sec(client: httpx.Client, url: str) -> dict[str, object]:
    if index_identity(url) is None:
        raise ContentError("SEC_INDEX_IDENTITY_INVALID")
    raw_index, _, resolved = fetch(client, url, MAX_HTML_BYTES)
    if resolved != url:
        raise ContentError("SEC_INDEX_REDIRECT")
    try:
        primary, exhibits = documents(raw_index, url)
    except ValueError as exc:
        raise ContentError(str(exc)) from exc
    if len(exhibits) > 8:
        raise ContentError("SEC_EXHIBIT_LIMIT_REVIEW")
    originals = []
    texts = []
    for target in (primary, *exhibits):
        time.sleep(0.2)
        raw, _, resolved = fetch(client, target, 8_000_000)
        if resolved != target:
            raise ContentError("SEC_DOCUMENT_REDIRECT")
        try:
            body = visible_text(raw, target)
        except ValueError as exc:
            raise ContentError("SEC_DOCUMENT_EXTRACTION_FAILED") from exc
        if classify(body) != "TEXT_CANDIDATE":
            raise ContentError("SEC_DOCUMENT_TEXT_INVALID")
        texts.append(body)
        originals.append({
            "url": target, "sha256": hashlib.sha256(raw).hexdigest(),
            "rawBase64": base64.b64encode(raw).decode("ascii"),
        })
    bundle = json.dumps({
        "schema": "sec_original_bundle_v1", "indexUrl": url,
        "indexSha256": hashlib.sha256(raw_index).hexdigest(),
        "indexRawBase64": base64.b64encode(raw_index).decode("ascii"),
        "documents": originals,
    }, sort_keys=True, separators=(",", ":")).encode()
    body = "\n\n".join(texts)
    return {
        "status": "COMPLETE", "body": body,
        "body_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "raw_sha256": hashlib.sha256(bundle).hexdigest(), "raw_payload": bundle,
        "source_url": url, "format": "SEC_BUNDLE", "sparse_pages": "[]",
    }


def capture(client: httpx.Client, url: str) -> dict[str, object]:
    # RSS can link to programmes, galleries and index pages rather than a
    # written article. Do not mislabel their template text as article content.
    if index_identity(url) is not None:
        return capture_sec(client, url)
    if media_page(url):
        raise ContentError("NON_ARTICLE_MEDIA_PAGE")
    payload, media_type, final_url = fetch(client, url, MAX_PDF_BYTES)
    if media_page(final_url):
        raise ContentError("NON_ARTICLE_MEDIA_PAGE")
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
        # Publisher release pages may contain only a short introduction. The
        # complete economic release or FOMC minutes can be in an official PDF.
        # Inspect explicitly linked, same-trust-boundary PDFs even if the HTML
        # is substantial; do not mistake a teaser for the full report.
        for pdf_url in pdf_links:
            try:
                candidate, _, resolved = fetch(client, pdf_url, MAX_PDF_BYTES)
                extracted, candidate_sparse = extract_pdf(candidate)
                if classify(extracted) == "TEXT_CANDIDATE" and len(extracted) > len(body):
                    body, sparse = extracted, candidate_sparse
                    source_data, source_url, kind = candidate, resolved, "PDF"
                    # Prefer the largest validated attachment, not an arbitrary
                    # minor table/appendix or the first PDF on the page.
            except (httpx.HTTPError, ContentError):
                continue
    else:
        raise ContentError("UNSUPPORTED_MEDIA_TYPE")
    quality = classify(body, url=source_url)
    if quality != "TEXT_CANDIDATE":
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
                    ('COMPLETE','REVIEW_REQUIRED','INCOMPLETE','NON_ARTICLE')),
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
    limit: int = MAX_PER_CYCLE, sec_only: bool = False,
) -> dict[str, object]:
    """Catch up on previously discovered RSS rows without mutating them."""
    if not 1 <= limit <= MAX_PER_CYCLE:
        raise ValueError("limit must be 1..12")
    store = NewsStore(db_path)
    initialize(store)
    now = datetime.now(UTC).isoformat()
    # Preserve capacity for three queues: new stories, historical catch-up,
    # and due retries. A high-volume feed must not starve retries forever.
    newest_slots = (limit + 2) // 3
    oldest_slots = (limit + 1) // 3
    retry_slots = limit - newest_slots - oldest_slots
    scope = "a.source GLOB 'sec_8k_[0-9]*'" if sec_only else "1=1"
    with store.connect() as db:
        base = """
            SELECT a.id, a.url, a.source FROM articles a
            LEFT JOIN article_full_text f ON f.article_id=a.id
            WHERE f.article_id IS NULL AND {scope}
              AND (a.source GLOB 'sec_8k_[0-9]*' OR a.source IN (
                'fed_monetary','fed_all','bok_all','bok_statistics',
                'bok_corrections','bbc_business','bbc_technology','bbc_world'
              ))
            ORDER BY a.id {order} LIMIT ?
        """
        newest = db.execute(base.format(order="DESC", scope=scope), (newest_slots,)).fetchall()
        oldest = db.execute(base.format(order="ASC", scope=scope), (oldest_slots,)).fetchall()
        retry_query = f"""
            SELECT a.id, a.url FROM articles a
            JOIN article_full_text f ON f.article_id=a.id
            WHERE f.status='INCOMPLETE' AND {scope}
              AND (f.retry_after <= ? OR (a.source GLOB 'sec_8k_[0-9]*'
                   AND f.error_code='INSUFFICIENT_ARTICLE_TEXT'))
              AND (a.source GLOB 'sec_8k_[0-9]*' OR a.source IN (
                'fed_monetary','fed_all','bok_all','bok_statistics',
                'bok_corrections','bbc_business','bbc_technology','bbc_world'
              ))
            ORDER BY f.retry_after ASC, a.id ASC LIMIT ?
        """
        retries = db.execute(retry_query, (now, retry_slots)).fetchall()
        chosen = {row[0]: row for row in (*newest, *oldest, *retries)}
        if len(chosen) < limit:
            # Refill unused retries or overlapping newest/oldest slots.
            for query, args in (
                (base.format(order="DESC", scope=scope), (limit,)),
                (base.format(order="ASC", scope=scope), (limit,)),
                (retry_query, (now, limit)),
            ):
                for row in db.execute(query, args).fetchall():
                    if len(chosen) >= limit:
                        break
                    chosen[row[0]] = row
                if len(chosen) >= limit:
                    break
        rows = list(chosen.values())
    owned = client is None
    client = client or httpx.Client(
        trust_env=False, follow_redirects=False,
        headers={"User-Agent": os.environ.get("MARKET_EVENT_SEC_USER_AGENT", USER_AGENT)},
    )
    counts = {"complete": 0, "review_required": 0, "incomplete": 0,
              "non_article": 0, "reused": 0}
    try:
        for article_id, url, source in rows:
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
                if existing and (not source.startswith("sec_8k_") or existing[6] == "SEC_BUNDLE") and classify(existing[5], url=url, digest=existing[2],
                                         sparse=existing[7]) == "TEXT_CANDIDATE":
                    row = dict(zip(("status","source_url","body_sha256",
                                    "raw_sha256","raw_payload","body",
                                    "format","sparse_pages"), existing))
                    counts["reused"] += 1
                else:
                    if source.startswith("sec_8k_") and owned and not os.environ.get(
                        "MARKET_EVENT_SEC_USER_AGENT", ""
                    ).strip():
                        raise ContentError("SEC_CONTACT_USER_AGENT_REQUIRED")
                    if source.startswith("sec_8k_"):
                        try:
                            index_url = canonical_index(url, source)
                        except ValueError as exc:
                            raise ContentError(str(exc)) from exc
                        row = capture_sec(client, index_url)
                    else:
                        row = capture(client, url)
            except (ContentError, httpx.HTTPError, OSError) as exc:
                error = str(exc) if isinstance(exc, ContentError) else type(exc).__name__
                state = ("NON_ARTICLE" if error == "NON_ARTICLE_MEDIA_PAGE"
                         else "INCOMPLETE")
                row = {
                    "status": state, "source_url": None,
                    "body_sha256": None, "raw_sha256": None,
                    "raw_payload": None, "body": None, "format": None,
                    "sparse_pages": "[]",
                }
            observed = datetime.now(UTC).isoformat()
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
        pending = db.execute(f"""
            SELECT COUNT(*) FROM articles a
            LEFT JOIN article_full_text f ON f.article_id=a.id
            WHERE {scope} AND (a.source GLOB 'sec_8k_[0-9]*' OR a.source IN (
                'fed_monetary','fed_all','bok_all','bok_statistics',
                'bok_corrections','bbc_business','bbc_technology','bbc_world'))
              AND (f.article_id IS NULL OR f.status IN ('INCOMPLETE','REVIEW_REQUIRED'))
        """).fetchone()[0]
    with store.connect() as db:
        review_pending = db.execute(f"""
            SELECT COUNT(*) FROM articles a JOIN article_full_text f
            ON f.article_id=a.id WHERE f.status='REVIEW_REQUIRED' AND {scope}
              AND (a.source GLOB 'sec_8k_[0-9]*' OR a.source IN (
                'fed_monetary','fed_all','bok_all','bok_statistics',
                'bok_corrections','bbc_business','bbc_technology','bbc_world'
              ))
        """).fetchone()[0]
        missing_or_failed = pending - review_pending
    return {
        "status": ("INCOMPLETE" if missing_or_failed else
                   "REVIEW_REQUIRED" if review_pending else "COMPLETE"),
        "attempted": len(rows), **counts,
        "pending_or_unverified": pending,
        "missing_or_failed": missing_or_failed,
        "review_pending": review_pending,
    }
