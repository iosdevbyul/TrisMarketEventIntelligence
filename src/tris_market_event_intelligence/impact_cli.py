"""Run a configurable local-first model on financial event hypotheses."""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import httpx

from tris_market_event_intelligence.impact import OpenAICompatibleProvider
from tris_market_event_intelligence.impact_diagnostics import inspect_errors
from tris_market_event_intelligence.impact_pilot import pilot_candidates
from tris_market_event_intelligence.impact_store import ImpactProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze news with a local or remote LLM")
    base = Path.home() / "Library" / "Application Support" / "TrisMarketEventIntelligence"
    parser.add_argument(
        "--db", type=Path, default=Path(os.getenv("MARKET_EVENT_DB", str(base / "news.sqlite3")))
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--live-freshness-minutes", type=int, default=None,
        help="Opt-in: defer old, undated or unhealthy source news before model analysis",
    )
    parser.add_argument(
        "--source-success-minutes", type=int, default=15,
        help="Max age of last successful source poll when using live freshness",
    )
    parser.add_argument("--provider", choices=("local", "remote"), default="local")
    parser.add_argument("--list-local-models", action="store_true")
    parser.add_argument("--show-errors", action="store_true",
                        help="Read saved analysis failures without invoking a model")
    parser.add_argument("--error-limit", type=int, default=10)
    parser.add_argument("--pilot-preview", action="store_true",
                        help="Read-only 3-event candidate preview; does not call Ollama")
    parser.add_argument("--event-ids", type=str, default=None,
                        help="Explicit comma-separated IDs (maximum 3) to analyze")
    parser.add_argument(
        "--timeout-seconds", type=int, default=None,
        help="Per-HTTP-operation timeout (local default: 300, remote default: 90; max: 900)",
    )
    args = parser.parse_args()
    if args.show_errors:
        try:
            result = inspect_errors(args.db.expanduser(), limit=args.error_limit)
        except ValueError as exc:
            parser.error(str(exc))
        print(json.dumps(result, ensure_ascii=False))
        if result['status'] in {'database_missing', 'database_unreadable'}:
            raise SystemExit(2)
        return
    if args.pilot_preview:
        try:
            report = pilot_candidates(
                args.db.expanduser(),
                model_id=os.getenv("MARKET_EVENT_LLM_MODEL", "gemma2:latest"),
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(json.dumps(report, ensure_ascii=False))
        if report["status"] != "ok":
            raise SystemExit(2)
        return
    selected_ids = None
    if args.event_ids is not None:
        try:
            raw = args.event_ids.split(",")
            if not 1 <= len(raw) <= 3 or any(
                not value.isascii() or not value.isdecimal() for value in raw
            ):
                raise ValueError("Expected 1..3 comma-separated positive integer IDs")
            selected_ids = tuple(int(value) for value in raw)
            if (
                any(value < 1 for value in selected_ids)
                or len(set(selected_ids)) != len(selected_ids)
                or args.batch_size < len(selected_ids)
            ):
                raise ValueError("IDs must be positive, distinct and within batch-size")
        except ValueError as exc:
            parser.error(str(exc))
    if args.list_local_models:
        try:
            response = httpx.get("http://127.0.0.1:11434/api/tags", timeout=5, trust_env=False)
            response.raise_for_status()
            for entry in response.json()["models"]:
                print(entry["name"])
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            parser.error(f"Local model server unavailable: {type(exc).__name__}")
        return

    if args.live_freshness_minutes is not None and not (
        1 <= args.live_freshness_minutes <= 1440
    ):
        parser.error("--live-freshness-minutes must be 1..1440")
    if not 1 <= args.source_success_minutes <= 1440:
        parser.error("--source-success-minutes must be 1..1440")
    model = os.getenv("MARKET_EVENT_LLM_MODEL")
    if not model:
        parser.error("Set MARKET_EVENT_LLM_MODEL to an installed model name")
    if args.provider == "local":
        base_url = "http://127.0.0.1:11434/v1"
        api_key = "local-placeholder"
        try:
            response = httpx.get(
                "http://127.0.0.1:11434/api/tags", timeout=5, trust_env=False
            )
            response.raise_for_status()
            available = {item["name"] for item in response.json()["models"]}
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            parser.error(f"Ollama readiness failed: {type(exc).__name__}")
        if model not in available:
            parser.error("Requested local model is not installed")
    else:
        base_url = os.getenv("MARKET_EVENT_LLM_BASE_URL", "https://api.openai.com/v1")
        api_key = os.getenv("MARKET_EVENT_LLM_API_KEY")
        if not api_key:
            parser.error("Remote provider requires MARKET_EVENT_LLM_API_KEY")
    timeout_seconds = args.timeout_seconds
    if timeout_seconds is None:
        timeout_seconds = 300 if args.provider == "local" else 90
    if not 1 <= timeout_seconds <= 900:
        parser.error("--timeout-seconds must be between 1 and 900")
    provider = OpenAICompatibleProvider(
        base_url=base_url, api_key=api_key, model=model,
        timeout_seconds=timeout_seconds,
        structured_output=False,
        native_ollama=args.provider == "local",
    )
    try:
        stats = ImpactProcessor(args.db.expanduser(), provider).process_pending(
            args.batch_size, event_ids=selected_ids,
            live_freshness_seconds=(
                args.live_freshness_minutes * 60
                if args.live_freshness_minutes is not None else None
            ),
            source_success_seconds=args.source_success_minutes * 60,
        )
    except ValueError as exc:
        parser.error(str(exc))
    output = asdict(stats)
    if args.live_freshness_minutes is None:
        output.pop("deferred")  # Preserve the legacy three-field JSON CLI contract.
    print(json.dumps(output))


if __name__ == "__main__":
    main()
