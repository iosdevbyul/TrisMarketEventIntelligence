"""Live full-body smoke test on eight publisher articles, isolated from production.

Reads one recent article per active source from the existing SQLite database
through a read-only connection. Creates a temporary scratch DB with the same
source URLs, then invokes the production full-text collector on those URLs.
No changes are made to the operational DB or running LaunchAgents.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.collector import Article, NewsStore
from tris_market_event_intelligence.operations import OPERATIONAL_SOURCES
from tris_market_event_intelligence.source_article_content import collect_full_text


def latest_articles(source_db: Path) -> list[tuple]:
    if source_db.is_symlink() or not source_db.is_file():
        raise ValueError("Expected existing, non-symlink news DB")
    uri = source_db.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        # Preserve source boundaries. Use the last observed RSS row from each
        # source, not an artificially assembled article or unrelated test URL.
        rows = db.execute("""
            SELECT a.source, a.url, a.title, a.summary,
                   a.published_at, a.detected_at, a.external_id
            FROM articles a
            INNER JOIN (
                SELECT source, MAX(id) AS newest_id
                FROM articles GROUP BY source
            ) latest ON latest.newest_id = a.id
            WHERE a.source IN (?,?,?,?,?,?,?,?)
            ORDER BY a.source
        """, OPERATIONAL_SOURCES).fetchall()
    return rows


def verify(source_db: Path) -> dict:
    rows = latest_articles(source_db)
    results = []
    with tempfile.TemporaryDirectory(prefix="tris-news-live-verify-") as root:
        scratch = NewsStore(Path(root) / "test.sqlite3")
        scratch.initialize()
        timestamp = datetime.now(UTC).isoformat()
        for source, url, title, summary, published, detected, external_id in rows:
            scratch.save(
                source=source, feed_url="https://example.org/scratch-fixture",
                payload=(f"<verification source='{source}'/>").encode(),
                articles=[Article(
                    source=source, external_id=external_id, url=url,
                    title=title, summary=summary, published_at=published,
                    detected_at=detected,
                )],
                fetched_at=timestamp, etag=None, last_modified=None,
            )
        stats = collect_full_text(scratch.path, limit=12) if rows else {
            "status": "INCOMPLETE", "attempted": 0,
        }
        with scratch.connect() as db:
            for source, url, title, *_ in rows:
                item = db.execute("""
                    SELECT f.status, LENGTH(f.body), f.format, f.error_code,
                           f.body, f.body_sha256, f.raw_sha256, f.raw_payload
                    FROM articles a JOIN article_full_text f ON f.article_id=a.id
                    WHERE a.source=? AND a.url=?
                """, (source, url)).fetchone()
                if item is None:
                    results.append({"source": source, "title": title, "url": url,
                                    "status": "NOT_ATTEMPTED"})
                    continue
                status, count, fmt, error, body, body_sha, raw_sha, payload = item
                verified = bool(body and payload and
                    hashlib.sha256(body.encode()).hexdigest() == body_sha and
                    hashlib.sha256(payload).hexdigest() == raw_sha)
                results.append({
                    "source": source, "title": title, "url": url,
                    "status": status, "bodyCharacters": count or 0,
                    "format": fmt, "errorCode": error,
                    "rawAndTextHashesVerified": verified,
                })
    present = {item["source"] for item in results}
    missing = sorted(set(OPERATIONAL_SOURCES) - present)
    collected = [
        r for r in results if r["status"] in ("COMPLETE", "REVIEW_REQUIRED")
    ]
    failures = [
        r for r in results if r["status"] not in (
            "COMPLETE", "REVIEW_REQUIRED", "NON_ARTICLE"
        ) or (r["status"] in ("COMPLETE", "REVIEW_REQUIRED")
              and not r.get("rawAndTextHashesVerified"))
    ]
    reviews = [r for r in results if r["status"] == "REVIEW_REQUIRED"]
    if missing or failures:
        verdict = "INCOMPLETE"
    elif reviews:
        verdict = "REVIEW_REQUIRED"
    else:
        verdict = "PASS"
    return {
        "schemaVersion": "fulltext_live_verification_v2",
        "sourcesExpected": len(OPERATIONAL_SOURCES),
        "sourcesTested": len(results),
        "missingSources": missing,
        "bodyCompleteCount": sum(
            r["status"] == "COMPLETE" and r.get("rawAndTextHashesVerified")
            for r in results
        ),
        "verifiedRawBodyCount": sum(
            bool(r.get("rawAndTextHashesVerified")) for r in collected
        ),
        "nonArticleCount": sum(r["status"] == "NON_ARTICLE" for r in results),
        "needsReviewCount": len(reviews),
        "failedCount": len(failures),
        "status": verdict,
        "collectionStats": stats,
        "results": results,
        "productionDatabaseModified": False,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.db.expanduser())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
