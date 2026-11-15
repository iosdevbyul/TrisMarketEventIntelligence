"""Opt-in local NEWS materiality candidate; never run or install models by default."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tris_market_event_intelligence.news_materiality_local import (
    generate_materiality_hypotheses,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Research news-only event materiality locally")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preview", action="store_true")
    modes.add_argument("--run", action="store_true")
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--model", default="gemma2:latest")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.run and args.output is None:
            raise ValueError("Explicit new private output path required for real inference")
        result = generate_materiality_hypotheses(
            cohort_path=args.cohort, model_id=args.model,
            run=args.run, limit=args.limit,
        )
        if args.output is not None:
            path = args.output
            if path.exists() or path.is_symlink() or not path.parent.is_dir():
                raise ValueError("Create-only output in an existing ordinary directory required")
            flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(path, flags, 0o600)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    json.dump(result, stream, indent=2, ensure_ascii=False)
                    stream.write("\n")
            except BaseException:
                path.unlink(missing_ok=True)
                raise
    except (ValueError, TypeError, KeyError, OSError,
            json.JSONDecodeError) as exc:
        parser.exit(2, f"news-materiality-local: {type(exc).__name__}: {exc}\n")
    print(json.dumps({
        "status": result["status"], "selectedArticles": result["selectedArticles"],
        "modelCalls": result["modelCalls"],
        "modelImprovementValidated": False, "promotionAllowed": False,
        "output": str(args.output) if args.output is not None else None,
    }, ensure_ascii=False))
    if result["status"] == "RESEARCH_INFERENCE_FAILED":
        parser.exit(1)


if __name__ == "__main__":
    main()
