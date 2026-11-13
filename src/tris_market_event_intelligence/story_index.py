"""Conservative article grouping: exact canonical URL or exact distinctive title.

These are story candidates, NOT verified events. No semantic matching or LLM use.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from tris_market_event_intelligence.collector import NewsStore, utc_now

TRACKING = {"fbclid", "gclid", "mc_cid", "mc_eid", "ref", "source"}
COMMON = {"utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_term"}
WORD = re.compile(r"[0-9A-Za-z가-힣]+")


def canonical_url(url: str) -> str:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if not host:
        raise ValueError("Missing URL hostname")
    # Strip the most common tracking query params, keeping semantic query params.
    params = [
        (key, val) for key, val in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in TRACKING
        and key.lower() not in COMMON
        and not key.lower().startswith("utm_")
    ]
    authority = host if parts.port is None else f"{host}:{parts.port}"
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), authority, path, urlencode(sorted(params)), ""))


def title_fingerprint(title: str, published: str | None) -> str | None:
    words = WORD.findall(unicodedata.normalize("NFKC", title).lower())
    # Avoid merging common generic headlines or different-day recaps.
    if len(words) < 5 or len("".join(words)) < 24 or not published:
        return None
    try:
        day = datetime.fromisoformat(published).date().isoformat()
    except ValueError:
        return None
    normalized = " ".join(words)
    return hashlib.sha256(f"{day}:{normalized}".encode()).hexdigest()


@dataclass(frozen=True)
class IndexStats:
    indexed: int
    new_groups: int
    url_matches: int
    title_matches: int


class StoryIndex:
    def __init__(self, database: Path) -> None:
        self.store = NewsStore(database)

    def initialize(self) -> None:
        self.store.initialize()
        with self.store.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS story_groups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    grouping_version TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS story_members (
                    article_id INTEGER PRIMARY KEY REFERENCES articles(id),
                    story_id INTEGER NOT NULL REFERENCES story_groups(id),
                    canonical_url TEXT NOT NULL,
                    title_fingerprint TEXT,
                    match_reason TEXT NOT NULL,
                    indexed_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_story_url ON story_members(canonical_url);
                CREATE INDEX IF NOT EXISTS idx_story_title
                    ON story_members(title_fingerprint);
                CREATE INDEX IF NOT EXISTS idx_story_group ON story_members(story_id);
                """
            )

    def index_pending(self, batch_size: int = 1000) -> IndexStats:
        if not 1 <= batch_size <= 10000:
            raise ValueError("batch_size must be between 1 and 10000")
        self.initialize()
        indexed = groups = url_matches = title_matches = 0
        with self.store.connect() as db:
            pending = db.execute(
                """
                SELECT a.id, a.url, a.title, a.published_at
                FROM articles a LEFT JOIN story_members m ON a.id=m.article_id
                WHERE m.article_id IS NULL ORDER BY a.id LIMIT ?
                """, (batch_size,),
            ).fetchall()
            for article_id, url, title, published in pending:
                canonical = canonical_url(url)
                fingerprint = title_fingerprint(title, published)
                with db:
                    row = db.execute(
                        "SELECT story_id FROM story_members WHERE canonical_url=? "
                        "ORDER BY article_id LIMIT 1", (canonical,),
                    ).fetchone()
                    reason = "canonical_url"
                    if row:
                        url_matches += 1
                    elif fingerprint:
                        row = db.execute(
                            "SELECT story_id FROM story_members WHERE title_fingerprint=? "
                            "ORDER BY article_id LIMIT 1", (fingerprint,),
                        ).fetchone()
                        reason = "exact_title_same_day"
                        if row:
                            title_matches += 1
                    if not row:
                        cursor = db.execute(
                            "INSERT INTO story_groups(created_at, grouping_version) "
                            "VALUES (?, ?)", (utc_now(), "strict-v1"),
                        )
                        story_id = cursor.lastrowid
                        reason = "new_story"
                        groups += 1
                    else:
                        story_id = row[0]
                    db.execute(
                        """INSERT INTO story_members (
                            article_id, story_id, canonical_url, title_fingerprint,
                            match_reason, indexed_at
                        ) VALUES (?, ?, ?, ?, ?, ?)""",
                        (article_id, story_id, canonical, fingerprint, reason, utc_now()),
                    )
                    indexed += 1
        return IndexStats(indexed, groups, url_matches, title_matches)

    def list_groups(self, limit: int = 100) -> list[dict[str, object]]:
        self.initialize()
        with self.store.connect() as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                """
                SELECT g.id, COUNT(m.article_id) AS articles, MIN(m.indexed_at) AS first_seen,
                       GROUP_CONCAT(DISTINCT a.source) AS sources
                FROM story_groups g
                JOIN story_members m ON m.story_id=g.id
                JOIN articles a ON a.id=m.article_id
                GROUP BY g.id ORDER BY g.id DESC LIMIT ?
                """, (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
