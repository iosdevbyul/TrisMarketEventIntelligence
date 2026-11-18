"""Bounded opt-in market-pressure event discovery and model research."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

from tris_market_event_intelligence.llm_runtime import (
    LLMCapabilities,
    LLMSettings,
)
from tris_market_event_intelligence.market_event_discovery import (
    discover_and_analyze,
    recent_candidates,
)
from tris_market_event_intelligence.market_pressure_results_store import MarketPressureResultStore
from tris_market_event_intelligence.news_llm_factory import build_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--as-of-time", type=datetime.fromisoformat)
    parser.add_argument("--lookback-days", type=int, default=10)
    parser.add_argument("--max-groups", type=int, default=2)
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--persist", action="store_true",
                        help="Explicitly persist completed review-only results to existing DB")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()

    if args.preview:
        if args.persist:
            parser.error("--persist requires --run")
        candidates, inventory = recent_candidates(
            args.db, as_of=args.as_of, lookback_days=args.lookback_days,
            as_of_time=args.as_of_time,
        )
        print(json.dumps({
            "status": "READ_ONLY_CANDIDATE_PREVIEW",
            "inventory": inventory,
            "candidateSourceIds": [item["evidence"].sourceId for item in candidates],
            "modelCalls": 0,
        }, ensure_ascii=False))
        return

    if args.output is None:
        parser.error("--run requires --output")
    if args.output.exists() or args.output.is_symlink():
        parser.error("OUTPUT_ALREADY_EXISTS")
    if not 1 <= args.max_groups <= 3:
        parser.error("--max-groups must be 1..3")

    settings = replace(
        LLMSettings.from_environment(explicit=True),
        model_id=args.model,
        capabilities=LLMCapabilities(
            json_schema=True, thinking=False, context_window=8192,
            max_output_tokens=4096,
        ),
    )
    provider = build_runtime(settings, confirm_model=True)
    result = discover_and_analyze(
        args.db, as_of=args.as_of, provider=provider,
        lookback_days=args.lookback_days, max_groups=args.max_groups,
        as_of_time=args.as_of_time,
    )
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    stored = None
    if args.persist and result["events"]:
        store = MarketPressureResultStore(args.db)
        store.initialize()
        stored = store.save(result)
    print(json.dumps({
        "status": result["status"],
        "eventGroups": len(result["events"]),
        "modelCalls": result["modelCalls"],
        "outputCreated": True,
        "persisted": stored,
    }))


if __name__ == "__main__":
    main()
