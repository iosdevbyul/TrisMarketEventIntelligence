"""Read-only Mac and local Ollama readiness snapshot, without model downloads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

from tris_market_event_intelligence.gemma4_mac_capacity import inspect_gemma4_mac
from tris_market_event_intelligence.gemma4_news_comparison_cli import _private_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Observe Mac memory and local Ollama Gemma 4 readiness without model changes"
    )
    parser.add_argument("--gemma4", default="gemma4:e4b")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            report = inspect_gemma4_mac(
                models=("gemma2:latest", args.gemma4),
                client=client,
            )
        if args.output is not None:
            _private_output(args.output, report)
            print(json.dumps({
                "status": report["status"],
                "output": str(args.output),
                "inferenceMemoryFitVerified": False,
                "modelDownloadedOrInstalled": False,
            }, ensure_ascii=False))
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2))
    except (ValueError, TypeError, OSError, httpx.HTTPError,
            json.JSONDecodeError) as exc:
        parser.exit(2, f"gemma-mac-preflight: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
