"""Collection-only supervisor for public news, SEC and economic evidence."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from tris_market_event_intelligence.collector import NewsStore, OfficialNewsCollector, utc_now
from tris_market_event_intelligence.news_source_provider import PUBLISHERS
from tris_market_event_intelligence.operations import _write_status
from tris_market_event_intelligence.sec_operational_discovery_cli import DEFAULT_CIKS, discover
from tris_market_event_intelligence.source_article_content import collect_full_text


def news_job(path, batches):
    discovery = OfficialNewsCollector(
        NewsStore(path), feeds={key: item.feed for key, item in PUBLISHERS.items()},
    ).collect_once()
    captures = []
    for _ in range(batches):
        capture = collect_full_text(path, media_only=True, limit=12)
        captures.append(capture)
        if not capture["attempted"]:
            break
    healthy = all(row["status"] in {"ok", "not_modified"} for row in discovery.values())
    return {"status": "OK" if healthy and not any(c["incomplete"] for c in captures)
            else "PARTIAL", "discovery": discovery, "captures": captures}


def sec_job(path, ciks, days, batches):
    agent = os.environ.get("MARKET_EVENT_SEC_USER_AGENT", "")
    if "@" not in agent:
        return {"status": "BLOCKED", "error": "SEC_CONTACT_USER_AGENT_REQUIRED"}
    end = datetime.now(UTC).date()
    store = NewsStore(path)
    results = []
    captures = []
    with httpx.Client(timeout=30, trust_env=False, headers={"User-Agent": agent}) as client:
        for cik in ciks:
            try:
                results.append({"status": "OK", **discover(
                    store, client, cik=cik, start=end - timedelta(days=days), end=end,
                )})
            except (httpx.HTTPError, ValueError, KeyError, TypeError, OSError, sqlite3.Error) as exc:
                results.append({"cik": cik, "status": "FAILED", "error": type(exc).__name__})
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (403, 429):
                    return {"status": "PARTIAL", "discovery": results, "captures": captures,
                            "error": "SEC_ACCESS_OR_RATE_LIMIT", "retryNextCycle": True}
        for _ in range(batches):
            capture = collect_full_text(path, client=client, sec_only=True, limit=12)
            captures.append(capture)
            if not capture["attempted"] or capture["incomplete"]:
                break
    return {"status": "OK" if all(r["status"] == "OK" for r in results)
            and not any(c["incomplete"] for c in captures) else "PARTIAL",
            "discovery": results, "captures": captures, "issuerScope": list(ciks),
            "firstPublicTimeVerified": False}


def economic_job(args):
    output = args.output_dir / "economic-source-report.json"
    result = subprocess.run([
        sys.executable, "-m", "tris_market_event_intelligence.global_economic_collection_cli",
        "--news-db", str(args.news_db), "--macro-db", str(args.macro_db),
        "--operational", "--output", str(output),
    ], capture_output=True, text=True, timeout=7200, check=False)
    if result.returncode:
        return {"status": "FAILED", "exitCode": result.returncode}
    report = json.loads(result.stdout)
    problems = [r for r in report["results"]
                if r.get("status") in {"FAILED", "PARTIAL", "EMPTY", "RATE_LIMITED"}
                or r.get("exitCode", 0)]
    return {"status": "PARTIAL" if problems else "OK", "report": report}


def run_job(args, job):
    started = utc_now()
    _write_status(args.output_dir / (job + "-running.json"), {
        "job": job, "startedAt": started, "status": "RUNNING",
        "analysisPerformed": False, "modelCalls": 0,
    })
    try:
        if job == "news":
            report = news_job(args.news_db, args.batches)
        elif job == "sec":
            report = sec_job(args.news_db, args.cik or DEFAULT_CIKS, args.sec_days, args.batches)
        else:
            report = economic_job(args)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, sqlite3.Error,
            httpx.HTTPError, subprocess.SubprocessError) as exc:
        report = {"status": "FAILED", "error": type(exc).__name__}
    record = {"job": job, "startedAt": started, "finishedAt": utc_now(), **report,
              "analysisPerformed": False, "modelCalls": 0}
    _write_status(args.output_dir / (job + "-latest.json"), record)
    (args.output_dir / (job + "-running.json")).unlink(missing_ok=True)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--news-db", type=Path, required=True)
    parser.add_argument("--macro-db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--job", action="append", choices=("news", "sec", "economic"))
    parser.add_argument("--cik", type=int, action="append")
    parser.add_argument("--sec-days", type=int, default=7)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.batches <= 10 or not 1 <= args.sec_days <= 30:
        parser.error("Invalid batch or SEC day limit")
    if args.cik and (len(set(args.cik)) != len(args.cik)
                     or len(args.cik) > 50 or any(not 1 <= c <= 9999999999 for c in args.cik)):
        parser.error("Invalid issuer scope")
    if not args.news_db.is_file() or args.news_db.is_symlink():
        parser.error("Existing non-symlink news database required")
    reports = [args.output_dir / (job + "-latest.json") for job in ("news", "sec", "economic")]
    reports.append(args.output_dir / "economic-source-report.json")
    paths = [args.news_db, args.macro_db, *reports]
    if len({p.resolve() for p in paths}) != len(paths):
        parser.error("Distinct database and report paths required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = tuple(dict.fromkeys(args.job or ("news", "sec", "economic")))
    with ExitStack() as stack:
        for job in sorted(jobs):
            lock_path = args.news_db.resolve().with_name(
                args.news_db.resolve().name + ".unified-" + job + ".lock"
            )
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            lock = stack.enter_context(os.fdopen(fd, "w"))
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.exit(2, "Collection job already running: " + job + "\\n")
        for job in jobs:
            marker = args.output_dir / (job + "-running.json")
            if marker.exists():
                _write_status(args.output_dir / (job + "-interrupted.json"), {
                    "job": job, "status": "INTERRUPTED_PREVIOUS_RUN",
                    "detectedAt": utc_now(),
                })
                marker.unlink()
        if not args.watch:
            for job in jobs:
                run_job(args, job)
            return
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = [executor.submit(watch_job, args, job) for job in jobs]
            for future in futures:
                future.result()


INTERVALS = {"news": 300, "sec": 900, "economic": 3600}


def watch_job(args, job):
    while True:
        run_job(args, job)
        time.sleep(INTERVALS[job])


if __name__ == "__main__":
    main()
