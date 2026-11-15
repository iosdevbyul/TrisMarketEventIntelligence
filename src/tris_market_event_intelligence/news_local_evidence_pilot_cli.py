"""Local real-news evidence pilot: preview by default, explicit tiny model run."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tris_market_event_intelligence.news_local_evidence_pilot import (
    audit_local_news_pilot,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit 1..3 pinned real NEWS candidates with installed local Gemma"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.run and args.output is None:
            raise ValueError("Real inference needs an explicit private output file")
        if args.output is not None and (
            args.output.exists() or args.output.is_symlink()
            or not args.output.parent.is_dir() or args.output.parent.is_symlink()
        ):
            raise ValueError("Existing ordinary directory and new output file required")
        report = audit_local_news_pilot(
            cohort_path=args.cohort, sample_path=args.sample,
            model_id=args.model, run=args.run, limit=args.limit,
        )
        if args.output is not None:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(args.output, flags, 0o600)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    json.dump(report, stream, ensure_ascii=False, indent=2)
                    stream.write("\n")
            except BaseException:
                args.output.unlink(missing_ok=True)
                raise
        print(json.dumps({
            "status": report["status"],
            "modelId": report["modelId"],
            "selectedArticles": report["selectedArticles"],
            "modelCalls": report["modelCalls"],
            "counts": report["counts"],
            "independentNewsAccuracyMeasured": False,
            "modelImprovementValidated": False,
            "promotionAllowed": False,
            "outputCreated": args.output is not None,
        }, ensure_ascii=False))
    except (ValueError, TypeError, OSError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, f"news-local-evidence-pilot: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
