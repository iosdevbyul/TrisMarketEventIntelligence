"""Single explicitly invoked RSS -> event -> optional local Gemma cycle.

A dry run performs no network requests, model calls, or filesystem writes.
The local model path is never activated without --enable-local-analysis.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import httpx

from tris_market_event_intelligence.impact import OpenAICompatibleProvider
from tris_market_event_intelligence.integrated_pipeline import run_integrated_pass
from tris_market_event_intelligence.operations import OPERATIONAL_SOURCES
from tris_market_event_intelligence.session_operations import (
    load_session_calendar,
    market_session_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a bounded market-news collection, extraction and optional local analysis",
    )
    base = Path.home() / "Library" / "Application Support" / "TrisMarketEventIntelligence"
    parser.add_argument("--db", type=Path, default=base / "news.sqlite3")
    parser.add_argument("--status", type=Path, default=base / "integrated-last-run.json")
    parser.add_argument("--sources", default=",".join(OPERATIONAL_SOURCES))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--enable-local-analysis", action="store_true")
    parser.add_argument("--enable-ticker-review", action="store_true",
                        help="Explicitly capture a review-only ticker report for new news")
    parser.add_argument("--listing-bundle", type=Path,
                        help="Already downloaded, pinned real listings")
    parser.add_argument("--review-output-root", type=Path,
                        help="Existing private folder for create-only review results")
    parser.add_argument("--live-freshness-minutes", type=int, default=15)
    parser.add_argument("--source-success-minutes", type=int, default=15)
    parser.add_argument("--max-extraction", type=int, default=1000)
    parser.add_argument("--sessions", type=Path, default=None,
                        help="Reviewed session CSV activates market-running guard")
    parser.add_argument("--markets", default="XKRX,XNYS,XNAS",
                        help="Target exchanges with --sessions")
    args = parser.parse_args()
    if args.enable_ticker_review:
        if not args.listing_bundle or not args.review_output_root:
            parser.error("--enable-ticker-review requires a listing bundle and output root")
        if args.enable_local_analysis:
            parser.error("Ticker review cannot activate Ollama/local analysis")
    elif args.listing_bundle or args.review_output_root:
        parser.error("Ticker inputs require explicit --enable-ticker-review")
    sources = tuple(s.strip() for s in args.sources.split(","))
    if not sources or len(set(sources)) != len(sources) or (
        set(sources) - set(OPERATIONAL_SOURCES)
    ):
        parser.error("Select unique known operational RSS source IDs only")
    if not 1 <= args.live_freshness_minutes <= 1440 or not (
        1 <= args.source_success_minutes <= 1440
    ):
        parser.error("Source and freshness budgets must be 1..1440 minutes")
    if not 1 <= args.max_extraction <= 1000:
        parser.error("--max-extraction must be 1..1000")
    if args.sessions is None and args.markets != "XKRX,XNYS,XNAS":
        parser.error("--markets requires --sessions")
    targets = tuple(name.strip() for name in args.markets.split(","))
    try:
        calendar = load_session_calendar(args.sessions) if args.sessions else None
        plan = (
            market_session_plan(
                calendar, now=datetime.now(UTC), markets=targets, sources=sources,
            ) if calendar is not None else None
        )
    except (OSError, ValueError, TypeError) as exc:
        parser.error(f"Invalid reviewed sessions: {type(exc).__name__}")
    if args.dry_run:
        print(json.dumps({
            "status": "dry_run", "sources": list(sources),
            "network_requests": 0, "filesystem_writes": 0,
            "model_invoked": False,
            "local_model_requested": args.enable_local_analysis,
            "background_service_installed": False,
            "ticker_review_requested": args.enable_ticker_review,
            "session_plan": plan,
        }))
        return
    if plan is not None and plan["decision"] != "RUN":
        print(json.dumps({
            "status": "session_skipped" if plan["decision"] == "SKIP" else "blocked",
            "session_plan": plan, "network_requests": 0, "model_invoked": False,
        }, ensure_ascii=False))
        if plan["decision"] != "SKIP":
            raise SystemExit(2)
        return
    provider = None
    if args.enable_local_analysis:
        model_name = os.getenv("MARKET_EVENT_LLM_MODEL", "")
        if not model_name.strip():
            parser.error("Set MARKET_EVENT_LLM_MODEL for explicitly opted-in local analysis")
        try:
            response = httpx.get(
                "http://127.0.0.1:11434/api/tags", timeout=5, trust_env=False,
            )
            response.raise_for_status()
            models = {item["name"] for item in response.json()["models"]}
            if model_name not in models:
                parser.error("Requested local Ollama model is not installed")
        except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
            parser.error(f"Local Ollama readiness failed: {type(exc).__name__}")
        provider = OpenAICompatibleProvider(
            base_url="http://127.0.0.1:11434/v1",
            api_key="local-placeholder", model=model_name,
            timeout_seconds=300, structured_output=False, native_ollama=True,
        )
    report = run_integrated_pass(
        args.db.expanduser(), status_path=args.status.expanduser(),
        selected_sources=sources, model=provider,
        enable_local_analysis=args.enable_local_analysis,
        live_freshness_seconds=args.live_freshness_minutes * 60,
        source_success_seconds=args.source_success_minutes * 60,
        max_event_extraction=args.max_extraction,
        session_calendar=calendar, target_markets=targets,
        listing_bundle=args.listing_bundle.expanduser().resolve()
        if args.listing_bundle else None,
        review_output_root=args.review_output_root.expanduser().resolve()
        if args.review_output_root else None,
    )
    print(json.dumps(report, ensure_ascii=False))
    if report["status"] not in {"healthy"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
