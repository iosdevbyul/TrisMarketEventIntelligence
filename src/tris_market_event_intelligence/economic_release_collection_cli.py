"""Collect official economic releases only, using the existing evidence store."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tris_market_event_intelligence.collector import NewsStore, OfficialNewsCollector
from tris_market_event_intelligence.source_article_content import (
    MAX_PER_CYCLE,
    collect_full_text,
    initialize,
)
from tris_market_event_intelligence.source_catalog import DEFAULT_FEEDS

KOREA_SOURCES = ("bok_all", "bok_statistics", "bok_corrections")
US_SOURCES = ("fed_monetary", "fed_all")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--region", choices=("korea", "us", "all"), default="korea")
    parser.add_argument("--max-batches", type=int, default=10)
    parser.add_argument("--skip-discovery", action="store_true")
    parser.add_argument("--ocr-db", type=Path, help="Opt-in local sparse-page recovery database")
    parser.add_argument("--ocr-limit", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.max_batches <= 1000:
        parser.error("max-batches must be 1..1000")
    sources = {
        "korea": KOREA_SOURCES,
        "us": US_SOURCES,
        "all": KOREA_SOURCES + US_SOURCES,
    }[args.region]
    store = NewsStore(args.db)
    initialize(store)
    discovery = None
    if not args.skip_discovery:
        discovery = OfficialNewsCollector(
            store, feeds={source: DEFAULT_FEEDS[source] for source in sources}
        ).collect_once()
    batches = []
    for _ in range(args.max_batches):
        result = collect_full_text(
            args.db, limit=MAX_PER_CYCLE, official_sources=sources,
        )
        batches.append(result)
        if result["attempted"] == 0 or result["missing_or_failed"] == 0:
            break
    placeholders = ",".join("?" for _ in sources)
    with store.connect() as db:
        rows = db.execute(
            f"""SELECT a.source, COALESCE(f.status,'NOT_CAPTURED'), COUNT(*),
                SUM(CASE WHEN a.published_at IS NULL THEN 1 ELSE 0 END)
                FROM articles a LEFT JOIN article_full_text f ON f.article_id=a.id
                WHERE a.source IN ({placeholders})
                GROUP BY a.source, COALESCE(f.status,'NOT_CAPTURED')""",
            sources,
        ).fetchall()
    coverage = [
        {"source": source, "status": status, "count": count,
         "missingPublishedAt": missing}
        for source, status, count, missing in rows
    ]
    recovery = None
    if args.ocr_db is not None:
        from tris_market_event_intelligence.economic_pdf_recovery_cli import recover

        recovery = recover(args.db, args.ocr_db, limit=args.ocr_limit)
    print(json.dumps({
        "discovery": discovery, "batches": batches, "coverage": coverage,
        "ocrRecovery": recovery,
        "analysisPerformed": False, "firstPublicAvailabilityVerified": False,
        "allAttachmentsCaptured": False,
    }, ensure_ascii=False))
    if recovery is not None and (recovery["failed"] or recovery["remainingPages"]):
        raise SystemExit(1)
    if (discovery is not None and any(
        result["status"] not in ("ok", "not_modified") for result in discovery.values()
    )) or not coverage or any(row["status"] != "COMPLETE" for row in coverage):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
