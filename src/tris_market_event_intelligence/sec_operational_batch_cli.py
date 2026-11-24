"""Finite SEC backlog processing and cached exhibit diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

from tris_market_event_intelligence.automatic_news_pipeline import process
from tris_market_event_intelligence.source_article_content import collect_full_text

SCOPE = "(a.source GLOB 'sec_8k_[0-9]*' OR a.source GLOB 'sec_filing_[0-9]*')"


def audit(path: Path, model: str) -> dict:
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        rows = db.execute(f"""
            SELECT a.id, f.status, f.format, f.error_code, n.status, n.error_code
            FROM articles a LEFT JOIN article_full_text f ON a.id=f.article_id
            LEFT JOIN automatic_news_analyses n ON a.id=n.article_id
                AND n.body_sha256=f.body_sha256 AND n.model=?
            WHERE {SCOPE} ORDER BY a.id
        """, (model,)).fetchall()
        exhibits = []
        for article_id, raw, digest in db.execute(f"""
            SELECT a.id, f.raw_payload, f.raw_sha256
            FROM articles a JOIN article_full_text f ON a.id=f.article_id
            WHERE {SCOPE} AND f.format='SEC_BUNDLE_PARTIAL'
        """):
            if not isinstance(raw, bytes) or hashlib.sha256(raw).hexdigest() != digest:
                exhibits.append({"articleId": article_id, "issue": "BUNDLE_DIGEST_INVALID"})
                continue
            for doc in json.loads(raw).get("documents", []):
                if doc["quality"] != "TEXT_CANDIDATE":
                    exhibits.append({"articleId": article_id, "url": doc["url"],
                                     "role": doc["role"], "quality": doc["quality"],
                                     "sha256": doc["sha256"]})
        return {
            "discovered": len(rows),
            "primaryReady": sum(row[1] == "COMPLETE" for row in rows),
            "fullSelectedBundle": sum(row[2] == "SEC_BUNDLE" for row in rows),
            "currentAnalysisCompleted": sum(row[4] == "COMPLETED" for row in rows),
            "pending": [{"articleId": row[0], "captureStatus": row[1],
                         "format": row[2], "captureError": row[3],
                         "analysisStatus": row[4], "analysisError": row[5]}
                        for row in rows if row[1] != "COMPLETE"
                        or row[2] != "SEC_BUNDLE" or row[4] != "COMPLETED"],
            "exhibitDiagnostics": exhibits,
        }


def drain(path: Path, model: str, *, max_batches: int, limit: int = 12,
          repair: bool = False) -> dict:
    if not 1 <= max_batches <= 30 or not 1 <= limit <= 12:
        raise ValueError("Invalid bounded SEC batch settings")
    completed = 0
    failed = 0
    stop = "BATCH_LIMIT"
    repair_ids = []
    if repair:
        repair_ids = [row["articleId"] for row in audit(path, model)["pending"]
                      if row["captureStatus"] != "COMPLETE"
                      or row["format"] == "SEC_BUNDLE_PARTIAL"]
    for index in range(max_batches):
        if repair:
            ids = tuple(repair_ids[index * limit:(index + 1) * limit])
            if not ids:
                stop = "REPAIR_PASS_FINISHED"
                break
            capture = collect_full_text(path, limit=limit, sec_only=True, repair_ids=ids)
        else:
            capture = collect_full_text(path, limit=limit, sec_only=True)
        analysis = process(path, model=model, limit=limit, collect=False, sec_only=True)
        print(json.dumps({"batch": index + 1, "capture": capture, "analysis": analysis}),
              flush=True)
        if analysis["status"] == "ALREADY_RUNNING":
            stop = "ANALYSIS_LOCKED"
            break
        completed += analysis["completed"]
        failed += analysis["failed"]
        if not capture["attempted"] and not analysis["results"]:
            stop = "NO_READY_WORK"
            break
        if (not repair and capture["attempted"] and not capture["complete"]
                and not analysis["completed"]):
            stop = "CAPTURE_STALLED"
            break
    return {"stopReason": stop, "completedThisRun": completed, "failedThisRun": failed,
            "coverage": audit(path, model), "marketPricesAnalyzed": False,
            "prospectiveForecasts": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument("--max-batches", type=int, default=15, choices=range(1, 31))
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--repair", action="store_true",
                        help="Retry each pending SEC source once, ignoring old retry timers")
    args = parser.parse_args()
    if not args.db.is_file() or args.db.is_symlink():
        parser.error("Existing non-symlink DB required")
    if args.run and "@" not in os.environ.get("MARKET_EVENT_SEC_USER_AGENT", ""):
        parser.error("Identifying MARKET_EVENT_SEC_USER_AGENT required")
    result = (drain(args.db, args.model, max_batches=args.max_batches, repair=args.repair)
              if args.run else audit(args.db, args.model))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
