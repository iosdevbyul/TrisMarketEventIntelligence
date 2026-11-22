"""Local read-only news intake and impact observability dashboard.

Run: python -m tris_market_event_intelligence.dashboard_server --db PATH
Served on loopback only. No collectors, model invocations, or DB writes.
"""
from __future__ import annotations

import argparse
import sqlite3
from contextlib import closing
from pathlib import Path

import uvicorn
from fastapi import HTTPException
from fastapi.responses import HTMLResponse

from tris_market_event_intelligence.api import create_app
from tris_market_event_intelligence.dashboard_page import PAGE


def snapshot(db_path: Path, *, limit: int = 100, source: str | None = None) -> dict:
    """Read existing observations without mutating or initializing the database."""
    if not db_path.is_file():
        raise FileNotFoundError("News database does not exist")
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        names = {r[0] for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        if "articles" not in names:
            raise ValueError("Article collection tables not initialized")

        def count(table: str) -> int | None:
            return int(db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) if table in names else None

        totals = {key: count(table) for key, table in (
            ("articles", "articles"), ("captures", "captures"),
            ("events", "economic_events"), ("impacts", "market_impacts"),
            ("analysisRuns", "impact_runs"), ("claims", "article_event_claims"),
        )}
        latest = db.execute(
            "SELECT MAX(detected_at) FROM articles"
        ).fetchone()[0]
        source_rows = []
        if "fetch_state" in names:
            source_rows = [dict(r) for r in db.execute(
                """SELECT s.source, s.last_checked_at AS lastCheckedAt,
                          s.last_success_at AS lastSuccessAt, s.last_error AS lastError,
                          (SELECT COUNT(*) FROM articles a WHERE a.source=s.source) AS articleCount
                   FROM fetch_state s ORDER BY s.source"""
            ).fetchall()]
        else:
            source_rows = [dict(r) for r in db.execute(
                """SELECT source, COUNT(*) AS articleCount,
                          NULL AS lastCheckedAt, NULL AS lastSuccessAt, NULL AS lastError
                   FROM articles GROUP BY source ORDER BY source"""
            ).fetchall()]
        for row in source_rows:
            row["recentAttempt"] = None
            if "fetch_attempts" in names:
                attempt = db.execute(
                    """SELECT status, completed_at AS completedAt,
                              returned, inserted FROM fetch_attempts
                       WHERE source=? ORDER BY id DESC LIMIT 1""", (row["source"],)
                ).fetchone()
                row["recentAttempt"] = dict(attempt) if attempt else None

        joins = []
        fields = [
            "a.id", "a.source", "a.url", "a.title", "a.summary",
            "a.published_at AS publishedAt", "a.detected_at AS detectedAt",
        ]
        if "article_full_text" in names:
            joins.append("LEFT JOIN article_full_text f ON f.article_id=a.id")
            fields.append("f.status AS fullTextStatus")
        else:
            fields.append("NULL AS fullTextStatus")
        if "extraction_runs" in names:
            fields.append("""(SELECT COUNT(*) FROM extraction_runs x
                             WHERE x.article_id=a.id) AS extractionCount""")
        else:
            fields.append("NULL AS extractionCount")
        if "economic_events" in names:
            fields.extend([
                """(SELECT COUNT(*) FROM economic_events e WHERE e.article_id=a.id)
                   AS eventCount""",
                """(SELECT e.status FROM economic_events e WHERE e.article_id=a.id
                   ORDER BY e.id DESC LIMIT 1) AS eventStatus""",
                """(SELECT e.classification FROM economic_events e WHERE e.article_id=a.id
                   ORDER BY e.id DESC LIMIT 1) AS eventClassification""",
            ])
            if "market_impacts" in names:
                fields.extend([
                    """(SELECT COUNT(*) FROM market_impacts i JOIN economic_events e
                       ON e.id=i.event_id WHERE e.article_id=a.id) AS impactCount""",
                    """(SELECT i.direction FROM market_impacts i JOIN economic_events e
                       ON e.id=i.event_id WHERE e.article_id=a.id
                       ORDER BY i.id DESC LIMIT 1) AS latestDirection""",
                    """(SELECT i.sector FROM market_impacts i JOIN economic_events e
                       ON e.id=i.event_id WHERE e.article_id=a.id
                       ORDER BY i.id DESC LIMIT 1) AS latestSector""",
                    """(SELECT i.rationale FROM market_impacts i JOIN economic_events e
                       ON e.id=i.event_id WHERE e.article_id=a.id
                       ORDER BY i.id DESC LIMIT 1) AS latestRationale""",
                    """(SELECT i.model_id FROM market_impacts i JOIN economic_events e
                       ON e.id=i.event_id WHERE e.article_id=a.id
                       ORDER BY i.id DESC LIMIT 1) AS latestModel""",
                ])
            else:
                fields.extend(["NULL AS impactCount", "NULL AS latestDirection",
                               "NULL AS latestSector", "NULL AS latestRationale",
                               "NULL AS latestModel"])
            if "impact_runs" in names:
                fields.append(
                    """(SELECT r.outcome FROM impact_runs r JOIN economic_events e
                       ON e.id=r.event_id WHERE e.article_id=a.id
                       ORDER BY r.rowid DESC LIMIT 1) AS latestRunOutcome"""
                )
            else:
                fields.append("NULL AS latestRunOutcome")
        else:
            fields.extend([
                "NULL AS eventCount", "NULL AS eventStatus",
                "NULL AS eventClassification", "NULL AS impactCount",
                "NULL AS latestDirection", "NULL AS latestSector",
                "NULL AS latestRationale", "NULL AS latestModel",
                "NULL AS latestRunOutcome",
            ])
        if "article_event_claims" in names:
            fields.append(
                "(SELECT COUNT(*) FROM article_event_claims c WHERE c.article_id=a.id) AS claimCount"
            )
        else:
            fields.append("NULL AS claimCount")

        where = "WHERE a.source=?" if source else ""
        params = (source, limit) if source else (limit,)
        rows = db.execute(
            f"SELECT {', '.join(fields)} FROM articles a {' '.join(joins)} "
            f"{where} ORDER BY a.id DESC LIMIT ?", params,
        ).fetchall()
        return {
            "schemaVersion": "dashboard_v1", "totals": totals,
            "lastArticleDetectedAt": latest, "sources": source_rows,
            "articles": [dict(row) for row in rows],
            "notes": {
                "scope": "Existing SQLite records only. No new collection or AI work is triggered.",
                "marketVerification": "Not available in this dashboard; predictions are not market-validated.",
                "analysis": "Impact fields reflect recorded market_impacts, not all independent Gemma batch artifacts.",
            },
        }


def create_dashboard_app(db_path: Path):
    app = create_app(db_path)

    @app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
    def dashboard():
        return HTMLResponse(PAGE, headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": (
                "default-src 'none'; script-src 'unsafe-inline'; "
                "style-src 'unsafe-inline'; connect-src 'self'; "
                "img-src 'none'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
            ),
        })

    @app.get("/api/v1/dashboard")
    def dashboard_data(limit: int = 100, source: str | None = None):
        if not 1 <= limit <= 300:
            raise HTTPException(status_code=422, detail="limit must be 1..300")
        if source is not None and len(source) > 100:
            raise HTTPException(status_code=422, detail="source too long")
        try:
            return snapshot(db_path, limit=limit, source=source)
        except (FileNotFoundError, ValueError, OSError, sqlite3.Error) as exc:
            raise HTTPException(status_code=503, detail="News data unavailable") from exc

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve local news observability dashboard")
    base = Path.home() / "Library/Application Support/TrisMarketEventIntelligence"
    parser.add_argument("--db", type=Path, default=base / "news.sqlite3")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be 1..65535")
    uvicorn.run(create_dashboard_app(args.db.expanduser().resolve()),
                host="127.0.0.1", port=args.port, proxy_headers=False)


if __name__ == "__main__":
    main()
