"""Explicit, read-only comparison of two configured models on identical NEWS.

Does not alter the currently deployed model or infer human-gold accuracy.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import httpx

from tris_market_event_intelligence.gemma4_news_comparison import compare_news_models
from tris_market_event_intelligence.gemma4_news_comparison_cli import _private_output
from tris_market_event_intelligence.llm_runtime import LLMCapabilities, LLMSettings
from tris_market_event_intelligence.news_llm_factory import build_impact_provider


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare any two explicit LLM models on same NEWS")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", action="store_true")
    mode.add_argument("--run", action="store_true")
    root = Path.home() / "Library/Application Support/TrisMarketEventIntelligence"
    parser.add_argument("--db", type=Path, default=root / "news.sqlite3")
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--kind-a", choices=("ollama", "openai-compatible"),
                        default="ollama")
    parser.add_argument("--kind-b", choices=("ollama", "openai-compatible"),
                        default="ollama")
    parser.add_argument("--endpoint-a", default=None)
    parser.add_argument("--endpoint-b", default=None)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        models = (args.model_a, args.model_b)
        providers = None
        if args.run:
            if args.output is None:
                raise ValueError("--run requires a new private --output file")
            settings = []
            for kind, model, endpoint in zip(
                (args.kind_a, args.kind_b), models,
                (args.endpoint_a, args.endpoint_b), strict=True,
            ):
                if kind == "ollama":
                    endpoint = endpoint or "http://127.0.0.1:11434"
                    key = ""
                else:
                    endpoint = endpoint or "https://api.openai.com/v1"
                    key = os.getenv("MARKET_EVENT_LLM_API_KEY", "")
                settings.append(LLMSettings(
                    kind=kind, model_id=model, endpoint=endpoint, api_key=key,
                    capabilities=LLMCapabilities(json_schema=True),
                    timeout_seconds=300 if kind == "ollama" else 90,
                ))
            providers = tuple(
                build_impact_provider(setting, confirm_model=True)
                for setting in settings
            )
        report = compare_news_models(
            db_path=args.db.expanduser(), models=models,
            analyzers=providers, limit=args.limit,
        )
        report["providerKinds"] = [args.kind_a, args.kind_b]
        report["modelSelectionExplicit"] = True
        if args.output is not None:
            _private_output(args.output, report)
            print(json.dumps({
                "status": report["status"],
                "countsByModel": report["countsByModel"],
                "selectedNewsEvents": report["selectedNewsEvents"],
                "modelCallCount": report["modelCallCount"],
                "outputCreated": True,
                "productionModelChanged": False,
                "promotionAllowed": False,
            }, ensure_ascii=False))
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2))
    except (ValueError, TypeError, KeyError, OSError, httpx.HTTPError) as exc:
        parser.exit(2, f"news-model-compare: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
