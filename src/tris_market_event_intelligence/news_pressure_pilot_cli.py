"""Explicit preview or bounded real Gemma news-pressure pilot, never persist."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

from tris_market_event_intelligence.llm_runtime import LLMSettings
from tris_market_event_intelligence.news_llm_factory import build_pressure_provider, with_model
from tris_market_event_intelligence.news_pressure_pilot import run_pressure_pilot
from tris_market_event_intelligence.news_pressure_provider import GemmaNewsPressureProvider


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-shot, read-only pilot of previously stored market impacts"
    )
    base = Path.home() / "Library/Application Support/TrisMarketEventIntelligence"
    parser.add_argument("--db", type=Path, default=base / "news.sqlite3")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--model", default=None)
    parser.add_argument("--configured-llm", action="store_true",
                        help="Opt in to configurable local/remote LLM execution")
    parser.add_argument("--run", action="store_true",
                        help="Explicit opt-in to call local Ollama at 127.0.0.1:11434")
    parser.add_argument("--output", type=Path,
                        help="New private report path; never overwrites existing file")
    args = parser.parse_args()
    try:
        if args.run and args.output is None:
            raise ValueError("--run requires --output to preserve review evidence")
        if args.output is not None and args.output.is_symlink():
            raise ValueError("Output symlink prohibited")
        analyzer = None
        if args.run and args.configured_llm:
            settings = LLMSettings.from_environment(explicit=True)
            if args.model is not None:
                settings = with_model(settings, args.model)
            analyzer = build_pressure_provider(settings, confirm_model=True)
        elif args.run:
            with httpx.Client(timeout=5, trust_env=False) as client:
                response = client.get("http://127.0.0.1:11434/api/tags")
                response.raise_for_status()
                installed = {item["name"] for item in response.json()["models"]}
            if (args.model or "gemma2:latest") not in installed:
                raise ValueError("Requested Gemma/Ollama model not installed")
            analyzer = GemmaNewsPressureProvider(model=args.model or "gemma2:latest")
        report = run_pressure_pilot(
            args.db.expanduser(), analyzer=analyzer, limit=args.limit,
        )
        if args.output is None:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            # Output is intentionally outside the running SQLite database.
            # Never replace a previous human-review pilot artifact.
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            print(json.dumps({
                "status": report["status"], "report": str(args.output),
                "selected": report["selected"], "counts": report["counts"],
                "model_call_attempts": report["model_call_attempts"],
                "production_database_modified": False,
            }, ensure_ascii=False))
    except (ValueError, OSError, httpx.HTTPError, KeyError, TypeError) as exc:
        parser.exit(2, f"news-pressure-pilot: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
