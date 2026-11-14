"""Official RSS ingestion with durable raw evidence and idempotent article storage."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx

MAX_FEED_BYTES = 2_000_000
USER_AGENT = "TrisMarketEventIntelligence/0.1 (+local research)"
from tris_market_event_intelligence.source_catalog import DEFAULT_FEEDS, validate_feeds


def safe_error_code(error: Exception) -> str:
    """Expose HTTP status without leaking query URLs or credentials."""
    if isinstance(error, httpx.HTTPStatusError):
        return f"HTTP_{error.response.status_code}"
    return type(error).__name__


FEEDS = DEFAULT_FEEDS


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class Article:
    source: str
    external_id: str
    url: str
    title: str
    summary: str | None
    published_at: str | None
    detected_at: str


def _text(node: ET.Element, name: str) -> str:
    child = node.find(name)
    return (child.text or "").strip() if child is not None else ""


def _parse_date(value: str) -> str | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC).isoformat()


def parse_rss(source: str, data: bytes, detected_at: str) -> list[Article]:
    """Parse RSS 2.0 items and Atom entries; reject invalid or oversized XML."""
    if len(data) > MAX_FEED_BYTES:
        raise ValueError("RSS response exceeds maximum size")
    root = ET.fromstring(data)
    articles: list[Article] = []
    for item in root.findall(".//item"):
        url = _text(item, "link")
        title = _text(item, "title")
        guid = _text(item, "guid")
        if not title or not url or urlparse(url).scheme not in {"http", "https"}:
            continue
        articles.append(
            Article(
                source=source,
                external_id=guid or url,
                url=url,
                title=title,
                summary=_text(item, "description") or None,
                published_at=_parse_date(_text(item, "pubDate")),
                detected_at=detected_at,
            )
        )
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for item in root.findall(".//a:entry", ns):
        title = _text(item, "{http://www.w3.org/2005/Atom}title")
        identifier = _text(item, "{http://www.w3.org/2005/Atom}id")
        link = item.find("a:link[@rel='alternate']", ns)
        if link is None:
            link = item.find("a:link", ns)
        url = (link.get("href") or "").strip() if link is not None else ""
        if not title or not url or urlparse(url).scheme not in {"http", "https"}:
            continue
        published = _text(item, "{http://www.w3.org/2005/Atom}published")
        updated = _text(item, "{http://www.w3.org/2005/Atom}updated")
        articles.append(
            Article(
                source=source,
                external_id=identifier or url,
                url=url,
                title=title,
                summary=_text(item, "{http://www.w3.org/2005/Atom}summary") or None,
                published_at=_parse_date(published or updated),
                detected_at=detected_at,
            )
        )
    return articles


class NewsStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def initialize(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS captures (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  source TEXT NOT NULL,
                  feed_url TEXT NOT NULL,
                  fetched_at TEXT NOT NULL,
                  sha256 TEXT NOT NULL,
                  payload BLOB NOT NULL,
                  UNIQUE(source, sha256)
                );
                CREATE TABLE IF NOT EXISTS articles (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  source TEXT NOT NULL,
                  external_id TEXT NOT NULL,
                  url TEXT NOT NULL,
                  title TEXT NOT NULL,
                  summary TEXT,
                  published_at TEXT,
                  detected_at TEXT NOT NULL,
                  capture_id INTEGER NOT NULL REFERENCES captures(id),
                  UNIQUE(source, external_id)
                );
                CREATE TABLE IF NOT EXISTS fetch_attempts (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  source TEXT NOT NULL,
                  request_started_at TEXT NOT NULL,
                  response_at TEXT,
                  completed_at TEXT NOT NULL,
                  request_elapsed_ms REAL,
                  status TEXT NOT NULL CHECK(status IN
                    ('ok', 'not_modified', 'error')),
                  returned INTEGER NOT NULL DEFAULT 0,
                  inserted INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_fetch_attempts_source_time
                  ON fetch_attempts(source, completed_at);
                CREATE TABLE IF NOT EXISTS fetch_state (
                  source TEXT PRIMARY KEY,
                  etag TEXT,
                  last_modified TEXT,
                  last_checked_at TEXT NOT NULL,
                  last_success_at TEXT,
                  last_error TEXT
                );
                """
            )

    def state(self, source: str) -> tuple[str | None, str | None]:
        with self.connect() as db:
            row = db.execute(
                "SELECT etag, last_modified FROM fetch_state WHERE source = ?", (source,)
            ).fetchone()
        return (row[0], row[1]) if row else (None, None)

    def record_error(self, source: str, error: str, checked_at: str) -> None:
        with self.connect() as db:
            db.execute(
                """
                INSERT INTO fetch_state(source, last_checked_at, last_error)
                VALUES (?, ?, ?)
                ON CONFLICT(source) DO UPDATE SET
                  last_checked_at=excluded.last_checked_at,
                  last_error=excluded.last_error
                """,
                (source, checked_at, error[:400]),
            )

    def save(
        self,
        source: str,
        feed_url: str,
        payload: bytes | None,
        articles: list[Article],
        fetched_at: str,
        etag: str | None,
        last_modified: str | None,
    ) -> int:
        """Atomically persist evidence, articles and successful fetch checkpoint."""
        inserted = 0
        with self.connect() as db:
            capture_id = None
            if payload is not None:
                digest = hashlib.sha256(payload).hexdigest()
                db.execute(
                    """
                    INSERT OR IGNORE INTO captures(source, feed_url, fetched_at, sha256, payload)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (source, feed_url, fetched_at, digest, payload),
                )
                capture_id = db.execute(
                    "SELECT id FROM captures WHERE source=? AND sha256=?", (source, digest)
                ).fetchone()[0]
                for a in articles:
                    cursor = db.execute(
                        """
                        INSERT OR IGNORE INTO articles
                        (source, external_id, url, title, summary, published_at,
                         detected_at, capture_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (a.source, a.external_id, a.url, a.title, a.summary,
                         a.published_at, a.detected_at, capture_id),
                    )
                    inserted += cursor.rowcount
            db.execute(
                """
                INSERT INTO fetch_state
                  (source, etag, last_modified, last_checked_at, last_success_at, last_error)
                VALUES (?, ?, ?, ?, ?, NULL)
                ON CONFLICT(source) DO UPDATE SET
                  etag=COALESCE(excluded.etag, fetch_state.etag),
                  last_modified=COALESCE(excluded.last_modified, fetch_state.last_modified),
                  last_checked_at=excluded.last_checked_at,
                  last_success_at=excluded.last_success_at,
                  last_error=NULL
                """,
                (source, etag, last_modified, fetched_at, fetched_at),
            )
        return inserted

    def record_attempt(
        self, source: str, started_at: str, response_at: str | None,
        completed_at: str, request_elapsed_ms: float | None, status: str,
        returned: int, inserted: int,
    ) -> None:
        """Persist fetch timing without source URLs or credentials."""
        with self.connect() as db:
            db.execute(
                """INSERT INTO fetch_attempts (
                  source,request_started_at,response_at,completed_at,
                  request_elapsed_ms,status,returned,inserted
                ) VALUES (?,?,?,?,?,?,?,?)""",
                (source, started_at, response_at, completed_at,
                 request_elapsed_ms, status, returned, inserted),
            )

    def stats(self) -> dict[str, int]:
        with self.connect() as db:
            return {
                name: db.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                for name in ("articles", "captures")
            }


class OfficialNewsCollector:
    def __init__(
        self,
        store: NewsStore,
        client: httpx.Client | None = None,
        feeds: dict[str, str] | None = None,
    ) -> None:
        self.store = store
        self.client = client
        self.feeds = validate_feeds(feeds if feeds is not None else FEEDS)

    def collect_once(self) -> dict[str, dict[str, str | int]]:
        self.store.initialize()
        outcome: dict[str, dict[str, str | int]] = {}
        owned = self.client is None
        client = self.client or httpx.Client(
            timeout=15, follow_redirects=True,
            headers={"User-Agent": USER_AGENT, "Accept": "application/rss+xml, application/xml"},
        )
        try:
            for source, url in self.feeds.items():
                started_at = utc_now()
                started_mono = time.monotonic()
                response_at: str | None = None
                request_elapsed_ms: float | None = None
                returned = inserted = 0
                status = "error"
                etag, modified = self.store.state(source)
                headers: dict[str, str] = {}
                if etag:
                    headers["If-None-Match"] = etag
                if modified:
                    headers["If-Modified-Since"] = modified
                try:
                    response = client.get(url, headers=headers)
                    response_at = utc_now()
                    request_elapsed_ms = max(0.0, (time.monotonic() - started_mono) * 1000)
                    if response.status_code == 304:
                        self.store.save(source, url, None, [], response_at, etag, modified)
                        status = "not_modified"
                    else:
                        response.raise_for_status()
                        payload = response.content
                        if len(payload) > MAX_FEED_BYTES:
                            raise ValueError("RSS response exceeds maximum size")
                        articles = parse_rss(source, payload, response_at)
                        returned = len(articles)
                        inserted = self.store.save(
                            source, url, payload, articles, response_at,
                            response.headers.get("etag"), response.headers.get("last-modified"),
                        )
                        status = "ok"
                except (httpx.HTTPError, ValueError, ET.ParseError) as exc:
                    self.store.record_error(source, safe_error_code(exc), utc_now())
                completed_at = utc_now()
                self.store.record_attempt(
                    source, started_at, response_at, completed_at,
                    request_elapsed_ms, status, returned, inserted,
                )
                outcome[source] = {"status": status, "inserted": inserted}
        finally:
            if owned:
                client.close()
        return outcome


def run_collector(
    db_path: Path, interval_seconds: int, once: bool, feeds: dict[str, str] | None = None
) -> None:
    if interval_seconds < 30:
        raise ValueError("interval_seconds must be at least 30")
    collector = OfficialNewsCollector(NewsStore(db_path), feeds=feeds)
    while True:
        print(json.dumps(collector.collect_once(), ensure_ascii=False), flush=True)
        if once:
            return
        time.sleep(interval_seconds)
