"""Audit or explicitly export self-attested human NEWS review labels."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tris_market_event_intelligence.news_review_sample_bridge import (
    bridge_review_sample,
)


def _write_new(path: Path, data: dict) -> None:
    if (path.is_symlink() or path.exists() or not path.parent.is_dir()
            or path.parent.is_symlink()):
        raise ValueError("Outputs must be new files in existing ordinary directories")
    flags = os.O_CREAT | os.O_WRONLY | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(data, out, ensure_ascii=False, indent=2)
            out.write("\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="No automatic human gold: validate and explicitly export NEWS reviews"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--importance-template", type=Path, required=True)
    parser.add_argument("--materiality-template", type=Path, required=True)
    parser.add_argument("--importance-output", type=Path)
    parser.add_argument("--materiality-output", type=Path)
    args = parser.parse_args()
    try:
        if args.audit and (args.importance_output or args.materiality_output):
            raise ValueError("Audit mode never writes review outputs")
        if args.apply and not (args.importance_output and args.materiality_output):
            raise ValueError("Apply mode needs both explicit new output paths")
        if args.apply and args.importance_output == args.materiality_output:
            raise ValueError("Two different output paths required")
        report = bridge_review_sample(
            cohort_path=args.cohort,
            sample_path=args.sample,
            importance_template_path=args.importance_template,
            materiality_template_path=args.materiality_template,
        )
        if args.apply:
            if not (report["importanceHumanJudgments"]
                    or report["materialityHumanJudgments"]):
                raise ValueError("No real human judgments: refusing to export blank files")
            paths = (args.importance_output, args.materiality_output)
            if any(path.exists() or path.is_symlink() or not path.parent.is_dir()
                   or path.parent.is_symlink() for path in paths):
                raise ValueError("Both output files must be new and private")
            created = []
            try:
                _write_new(paths[0], report["importanceReview"])
                created.append(paths[0])
                _write_new(paths[1], report["materialityReview"])
                created.append(paths[1])
            except BaseException:
                for path in created:
                    path.unlink(missing_ok=True)
                raise
    except (ValueError, TypeError, KeyError, OSError,
            json.JSONDecodeError) as exc:
        parser.exit(2, f"news-review-bridge: {type(exc).__name__}: {exc}\n")
    summary = {k: report[k] for k in (
        "status", "selectedArticles", "importanceHumanJudgments",
        "materialityHumanJudgments", "humanStoryRelationsNotPropagated",
        "humanGoldIndependentlyAudited", "modelCalls",
        "testGoldScored", "modelImprovementValidated", "promotionAllowed",
        "stockPricesUsed",
    )}
    summary["outputFilesCreated"] = bool(args.apply)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
