"""Read existing source records without claiming independent causal verification."""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime, timedelta

from tris_market_event_intelligence.news_near_duplicate_audit import _tokens


def _clock(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("Source timestamp must include timezone")
    return result.astimezone(UTC)


def related_sources(db: sqlite3.Connection, article_id: int, cutoff: str) -> list[dict]:
    """Retrieve locally observed, hash-matched related news, filings and releases."""
    at = _clock(cutoff)
    original = db.execute(
        "SELECT title FROM articles WHERE id=?", (article_id,),
    ).fetchone()
    if original is None:
        raise ValueError("Original article missing")
    words = _tokens(original[0])
    rows = db.execute(
        """SELECT a.id, a.source, a.title, a.url, a.published_at, a.detected_at,
                  f.body, f.body_sha256, f.observed_at
           FROM articles a JOIN article_full_text f ON f.article_id=a.id
           WHERE a.id<>? AND f.status='COMPLETE' AND f.body IS NOT NULL
           ORDER BY a.id DESC""", (article_id,),
    )
    candidates = []
    for row in rows:
        try:
            published, detected, observed = map(_clock, (row[4], row[5], row[8]))
        except (TypeError, ValueError):
            continue
        if not at - timedelta(days=7) <= published <= detected <= at or observed > at:
            continue
        if observed < published or not isinstance(row[6], str):
            continue
        if hashlib.sha256(row[6].encode()).hexdigest() != row[7]:
            continue
        shared = words & _tokens(row[2])
        if len(shared) < 3:
            continue
        candidates.append({
            "articleId": row[0], "source": row[1], "title": row[2], "url": row[3],
            "publishedAt": row[4], "observedAt": row[8], "bodySha256": row[7],
            "excerpt": row[6][:1000], "sharedTokens": len(shared),
            "status": "RELATED_SOURCE_CANDIDATE_NOT_CAUSAL_EVIDENCE",
        })
    return sorted(candidates, key=lambda x: (-x["sharedTokens"], x["articleId"]))[:3]


def story_overlap(db: sqlite3.Connection, article_id: int, training_ids: list[int]) -> bool:
    """Conservatively block lexical overlap; not a full semantic deduplication."""
    row = db.execute("SELECT title, url FROM articles WHERE id=?", (article_id,)).fetchone()
    if row is None:
        raise ValueError("Validation article missing")
    title, url = row
    words = _tokens(title)
    for key in training_ids:
        prior = db.execute("SELECT title, url FROM articles WHERE id=?", (key,)).fetchone()
        if prior is None:
            raise ValueError("Training article missing")
        other = _tokens(prior[0])
        union = words | other
        if key == article_id or (url and url == prior[1]):
            return True
        if words and words == other:
            return True
        if union and len(words & other) >= 3 and len(words & other) / len(union) >= 0.65:
            return True
    return False
