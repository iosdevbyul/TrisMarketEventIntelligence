"""Run a configurable local-first model on financial event hypotheses."""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import httpx

from tris_market_event_intelligence.impact import OpenAICompatibleProvider
from tris_market_event_intelligence.impact_diagnostics import inspect_errors
from tris_market_event_intelligence.impact_store import ImpactProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze news with a local or remote LLM")
    base = Path.home() / "Library" / "Application Support" / "TrisMarketEventIntelligence"
    parser.add_argument(
        "--db", type=Path, default=Path(os.getenv("MARKET_EVENT_DB", str(base / "news.sqlite3")))
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--provider", choices=("local", "remote"), default="local")
    parser.add_argument("--list-local-models", action="store_true")
    parser.add_argument("--show-errors", action="store_true",
                        help="Read saved analysis failures without invoking a model")
    parser.add_argument("--error-limit", type=int, default=10)
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
    if args.list_local_models:
        try:
            response = httpx.get("http://127.0.0.1:11434/api/tags", timeout=5, trust_env=False)
            response.raise_for_status()
            for entry in response.json()["models"]:
                print(entry["name"])
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            parser.error(f"Local model server unavailable: {type(exc).__name__}")
        return

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
    )
    stats = ImpactProcessor(args.db.expanduser(), provider).process_pending(args.batch_size)
    print(json.dumps(asdict(stats)))


if __name__ == "__main__":
    main()
