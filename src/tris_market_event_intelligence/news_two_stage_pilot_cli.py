"""Manual, read-only local Ollama two-stage news analysis pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

from tris_market_event_intelligence.impact import OpenAICompatibleProvider
from tris_market_event_intelligence.news_pressure_provider import GemmaNewsPressureProvider
from tris_market_event_intelligence.news_two_stage_pilot import run_two_stage_pilot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview or run a maximum three-event news direction + pressure pilot"
    )
    base = Path.home() / "Library/Application Support/TrisMarketEventIntelligence"
    parser.add_argument("--db", type=Path, default=base / "news.sqlite3")
    parser.add_argument("--model", default="gemma2:latest")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.run and args.output is None:
            raise ValueError("--run requires a new private --output file")
        if args.output is not None and args.output.is_symlink():
            raise ValueError("Output symlinks are prohibited")
        direction = pressure = None
        if args.run:
            with httpx.Client(timeout=5, trust_env=False) as client:
                response = client.get("http://127.0.0.1:11434/api/tags")
                response.raise_for_status()
                names = {item["name"] for item in response.json()["models"]}
                if args.model not in names:
                    raise ValueError("Selected model is not installed in Ollama")
            direction = OpenAICompatibleProvider(
                base_url="http://127.0.0.1:11434/v1",
                api_key="local-placeholder", model=args.model,
                native_ollama=True, timeout_seconds=300,
            )
            pressure = GemmaNewsPressureProvider(model=args.model)
        report = run_two_stage_pilot(
            args.db.expanduser(), direction_model=direction,
            strength_model=pressure, limit=args.limit, model_id=args.model,
        )
        if args.output is None:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            with args.output.open("x", encoding="utf-8") as out:
                json.dump(report, out, ensure_ascii=False, indent=2)
                out.write("\n")
            print(json.dumps({
                "status": report["status"], "report": str(args.output),
                "selected_events": report["selected_events"],
                "direction_model_calls": report["direction_model_calls"],
                "strength_model_calls": report["strength_model_calls"],
                "counts": report["counts"],
            }, ensure_ascii=False))
    except (ValueError, OSError, httpx.HTTPError, KeyError, TypeError) as exc:
        parser.exit(2, f"news-two-stage-pilot: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
