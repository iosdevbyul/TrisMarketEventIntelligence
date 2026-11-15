"""Manual, local-only Gemma 4 comparison; never downloads model weights."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import httpx

from tris_market_event_intelligence.gemma4_news_comparison import (
    _validate_models,
    compare_gemma_news,
)
from tris_market_event_intelligence.impact import OpenAICompatibleProvider


def local_model_preflight(
    models: tuple[str, str], *, client: httpx.Client,
) -> dict[str, object]:
    """Read installed local Ollama tags only; no pull or remote HTTP."""
    _validate_models(models)
    response = client.get("http://127.0.0.1:11434/api/tags")
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise TypeError("Invalid local Ollama model-list response")
    installed = {
        entry["name"] for entry in payload["models"]
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    }
    return {
        "schemaVersion": "gemma4_local_preflight_v1",
        "models": list(models),
        "eachRequestedModelInstalled": all(model in installed for model in models),
        "missingModels": [model for model in models if model not in installed],
        "realMemoryHeadroomVerified": False,
        "realInferenceCompatibilityVerified": False,
        "modelDownloadedOrInstalled": False,
        "productionModelChanged": False,
    }


def _private_output(path: Path, report: dict[str, object]) -> None:
    if path.is_symlink() or path.exists() or not path.parent.is_dir():
        raise ValueError("Create-only private output in an existing directory required")
    encoded = (json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()
    if len(encoded) > 100000:
        raise ValueError("Model comparison report too large")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the same NEWS in installed local Gemma 2 and Gemma 4"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    root = Path.home() / "Library/Application Support/TrisMarketEventIntelligence"
    parser.add_argument("--db", type=Path, default=root / "news.sqlite3")
    parser.add_argument("--gemma4", default="gemma4:e4b")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    models = ("gemma2:latest", args.gemma4)
    try:
        _validate_models(models)
        if args.run and args.output is None:
            raise ValueError("Real model comparison requires create-only --output")
        if args.preflight:
            with httpx.Client(timeout=5, trust_env=False) as client:
                report = local_model_preflight(models, client=client)
        elif args.run:
            with httpx.Client(timeout=5, trust_env=False) as client:
                preflight = local_model_preflight(models, client=client)
            if not preflight["eachRequestedModelInstalled"]:
                raise ValueError("Requested local Ollama models not installed")
            providers = tuple(
                OpenAICompatibleProvider(
                    base_url="http://127.0.0.1:11434/v1",
                    api_key="local-placeholder", model=model,
                    native_ollama=True, timeout_seconds=300,
                ) for model in models
            )
            report = compare_gemma_news(
                db_path=args.db.expanduser(), models=models,
                analyzers=providers, limit=args.limit,
            )
        else:
            report = compare_gemma_news(
                db_path=args.db.expanduser(), models=models, limit=args.limit,
            )
        if args.output is not None:
            _private_output(args.output, report)
            print(json.dumps({
                "status": report["status"], "output": str(args.output),
                "promotionAllowed": False,
                "modelDownloadedOrInstalled": False,
            }, ensure_ascii=False))
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2))
    except (ValueError, TypeError, OSError, KeyError, httpx.HTTPError,
            json.JSONDecodeError) as exc:
        parser.exit(2, f"gemma4-news-comparison: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
