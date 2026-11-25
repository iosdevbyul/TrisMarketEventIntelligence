"""Read-only inventory of collected economic evidence; no network or analysis."""
from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from tris_market_event_intelligence.collector import utc_now
from tris_market_event_intelligence.global_economic_collection_cli import (
    BOARDS,
    DEFERRED_RELEASES,
    FEEDS,
    NOT_CONNECTED,
)


def query(path: Path, statements: dict[str, tuple[str, str]]) -> dict:
    if not path.is_file():
        return {"status": "MISSING_DATABASE", "tables": {}}
    result = {}
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for name, (table, sql) in statements.items():
            result[name] = (
                [dict(row) for row in db.execute(sql)] if table in tables else None
            )
    return {"status": "READ", "tables": result}


def inventory(news: Path, macro: Path, ocr: Path | None = None) -> dict:
    sources = sorted(FEEDS.keys() | BOARDS.keys() | {
        "bok_all", "bok_statistics", "bok_corrections", "fed_all", "fed_monetary",
        "mods_cpi", "mods_industry", "mods_all",
    })
    # Source names are fixed registry constants, not external input.
    scope = ",".join("'" + source + "'" for source in sources)
    numeric = query(macro, {
        "series": ("macro_observation", """
            SELECT source,series,COUNT(*) AS observations,COUNT(DISTINCT period) AS periods,
            MIN(period) AS first_period,MAX(period) AS last_period,
            SUM(CASE WHEN vintage_start='' THEN 1 ELSE 0 END) AS unknown_vintage
            FROM macro_observation GROUP BY source,series ORDER BY source,series
        """),
        "cooldowns": ("macro_source_cooldown", "SELECT * FROM macro_source_cooldown"),
    })
    releases = query(news, {
        "articles": ("articles", f"""
            SELECT source,COUNT(*) AS records,
            SUM(CASE WHEN published_at IS NULL OR published_at='' THEN 1 ELSE 0 END)
                AS missing_published_at,
            MIN(published_at) AS earliest_declared,MAX(published_at) AS latest_declared
            FROM articles WHERE source IN ({scope}) GROUP BY source ORDER BY source
        """),
        "capture_states": ("official_release_capture_state", """
            SELECT a.source,s.status,COUNT(*) AS records
            FROM official_release_capture_state s JOIN articles a ON a.id=s.article_id
            GROUP BY a.source,s.status ORDER BY a.source,s.status
        """),
        "documents": ("official_release_documents", """
            SELECT a.source,d.status,COUNT(*) AS records,COUNT(DISTINCT d.sha256) AS unique_hashes
            FROM official_release_documents d JOIN articles a ON a.id=d.article_id
            GROUP BY a.source,d.status ORDER BY a.source,d.status
        """),
        "date_evidence": ("official_release_date_evidence", """
            SELECT a.source,COUNT(DISTINCT e.article_id) AS articles,COUNT(*) AS evidence_rows
            FROM official_release_date_evidence e JOIN articles a ON a.id=e.article_id
            GROUP BY a.source ORDER BY a.source
        """),
        "legacy_text": ("article_full_text", f"""
            SELECT a.source,f.status,COUNT(*) AS records
            FROM article_full_text f JOIN articles a ON a.id=f.article_id
            WHERE a.source IN ({scope}) GROUP BY a.source,f.status ORDER BY a.source,f.status
        """),
    })
    observed = {row["source"] for row in releases["tables"].get("articles") or []}
    series = numeric["tables"].get("series") or []
    return {
        "observedAt": utc_now(), "numeric": numeric, "releases": releases,
        "numericSeriesCount": len(series),
        "numericObservationRows": sum(row["observations"] for row in series),
        "configuredSourcesWithoutArticles": sorted(set(sources) - observed),
        "deferredSources": sorted(DEFERRED_RELEASES), "notConnected": list(NOT_CONNECTED),
        "ocr": query(ocr, {"pages": ("economic_pdf_recovery", """
            SELECT status,COUNT(*) AS evidence_rows,COUNT(DISTINCT raw_sha256) AS documents
            FROM economic_pdf_recovery GROUP BY status ORDER BY status
        """)}) if ocr else None,
        "analysisPerformed": False, "firstPublicAvailabilityVerified": False,
        "allAttachmentsCaptured": False, "visualReviewRequired": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--news-db", type=Path, required=True)
    parser.add_argument("--macro-db", type=Path, required=True)
    parser.add_argument("--ocr-db", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.news_db, args.macro_db, args.output] + ([args.ocr_db] if args.ocr_db else [])
    if len({path.resolve() for path in paths}) != len(paths):
        parser.error("Database and report paths must be distinct")
    report = inventory(args.news_db, args.macro_db, args.ocr_db)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
