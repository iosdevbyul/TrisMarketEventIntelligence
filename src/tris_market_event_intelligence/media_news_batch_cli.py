"""Bounded BBC news backlog, read-only reconciliation and scoped recovery."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

from tris_market_event_intelligence.automatic_news_pipeline import process
from tris_market_event_intelligence.news_content_quality import classify, media_page
from tris_market_event_intelligence.operations import run_cycle
from tris_market_event_intelligence.source_article_content import collect_full_text

SOURCES = ("bbc_business", "bbc_technology", "bbc_world")
SCOPE = "a.source IN ('bbc_business','bbc_technology','bbc_world')"


def audit(path: Path, model: str) -> dict:
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        has_analysis = "automatic_news_analyses" in tables
        join = (
            "LEFT JOIN automatic_news_analyses n ON n.article_id=a.id "
            "AND n.body_sha256=f.body_sha256 AND n.model=?"
            if has_analysis else ""
        )
        fields = "n.status,n.error_code,n.attempts" if has_analysis else "NULL,NULL,NULL"
        counts: Counter[str] = Counter()
        by_source: dict[str, Counter[str]] = {}
        pending = []
        excluded = []
        for row in db.execute(f"""
            SELECT a.id,a.source,a.title,a.url,f.status,f.body,f.body_sha256,
                   f.raw_payload,f.raw_sha256,f.sparse_pages,f.error_code,{fields}
            FROM articles a LEFT JOIN article_full_text f ON f.article_id=a.id
            {join} WHERE {SCOPE} ORDER BY a.id
        """, (model,) if has_analysis else ()):
            aid, source, title, url, status, body, digest, raw, raw_hash, sparse, error, state, reason, attempts = row
            quality = classify(body, title=title, url=url, digest=digest, sparse=sparse)
            if media_page(url):
                category = "EXCLUDED_AUDIO_VIDEO"
            elif status is None:
                category = "NOT_CAPTURED"
            elif status == "NON_ARTICLE":
                category = "MEDIA_REVIEW_REQUIRED"
            elif status != "COMPLETE":
                category = "CAPTURE_REVIEW" if status == "REVIEW_REQUIRED" else "CAPTURE_FAILED"
            elif not digest or not isinstance(raw, bytes) or hashlib.sha256(raw).hexdigest() != raw_hash:
                category = "ORIGINAL_INTEGRITY_REVIEW"
            elif quality != "TEXT_CANDIDATE":
                category = quality
            elif state != "COMPLETED":
                category = "MODEL_FAILED" if state == "FAILED" else "READY_FOR_ANALYSIS"
            else:
                category = "ANALYZED"
            counts[category] += 1
            by_source.setdefault(source, Counter())[category] += 1
            if category == "EXCLUDED_AUDIO_VIDEO":
                excluded.append({"articleId": aid, "source": source, "url": url,
                                 "existingAnalysisCompleted": state == "COMPLETED"})
            elif category != "ANALYZED":
                pending.append({"articleId": aid, "source": source, "title": title,
                                "url": url, "category": category, "captureError": error,
                                "analysisError": reason, "attempts": attempts,
                                "existingAnalysisCompleted": state == "COMPLETED"})
        return {"discovered": sum(counts.values()), "counts": dict(counts),
                "bySource": by_source, "pending": pending,
                "excludedAudioVideo": excluded,
                "textScopeCount": sum(counts.values()) - len(excluded),
                "semanticCorrectnessVerified": False, "marketPricesAnalyzed": False}


def drain(path: Path, model: str, *, max_batches: int, repair: bool = False) -> dict:
    if not 1 <= max_batches <= 100:
        raise ValueError("Invalid batch bound")
    before = audit(path, model)
    repair_ids = [row["articleId"] for row in before["pending"]
                  if row["category"] in {"CAPTURE_FAILED", "CAPTURE_REVIEW"}]
    completed = failed = 0
    stop = "BATCH_LIMIT"
    for index in range(max_batches):
        if repair:
            ids = tuple(repair_ids[index * 12:(index + 1) * 12])
            if not ids:
                stop = "REPAIR_PASS_FINISHED"
                break
            capture = collect_full_text(path, media_only=True, limit=12, repair_ids=ids)
        else:
            capture = collect_full_text(path, media_only=True, limit=12)
        analysis = process(path, model=model, limit=24, collect=False, media_only=True)
        print(json.dumps({"batch": index + 1, "capture": capture, "analysis": analysis}),
              flush=True)
        if analysis["status"] == "ALREADY_RUNNING":
            stop = "ANALYSIS_LOCKED"
            break
        completed += analysis["completed"]
        failed += analysis["failed"]
        if not repair and not capture["attempted"] and not analysis["results"]:
            stop = "NO_READY_WORK"
            break
    return {"stopReason": stop, "completedThisRun": completed, "failedThisRun": failed,
            "coverage": audit(path, model), "prospectiveForecasts": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument("--max-batches", type=int, choices=range(1, 101), default=30)
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--retry-failed", action="store_true",
                        help="One bounded retry of previously failed text analyses")
    args = parser.parse_args()
    if not args.db.is_file() or args.db.is_symlink():
        parser.error("Existing non-symlink DB required")
    if args.repair and not args.run:
        parser.error("--repair requires --run")
    discovery = run_cycle(args.db, selected_sources=SOURCES) if args.discover else None
    if args.retry_failed:
        if args.run or args.repair or args.discover:
            parser.error("--retry-failed is a standalone operation")
        before = audit(args.db, args.model)
        ids = tuple(row["articleId"] for row in before["pending"]
                    if row["category"] == "MODEL_FAILED")[:24]
        result = {"retry": process(args.db, model=args.model, collect=False, media_only=True,
                                   limit=24, retry_failed_ids=ids) if ids else None,
                  "coverage": audit(args.db, args.model)}
    else:
        result = (drain(args.db, args.model, max_batches=args.max_batches, repair=args.repair)
                  if args.run else audit(args.db, args.model))
    print(json.dumps({"discovery": discovery, "result": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
