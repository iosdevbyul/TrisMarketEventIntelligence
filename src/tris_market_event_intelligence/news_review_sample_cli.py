"""Generate a private review-candidate packet; print only safe counts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tris_market_event_intelligence.news_review_sample import (
    prepare_news_review_sample,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NEWS-only candidate event grouping and human review sample"
    )
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--max-selected", type=int, default=48)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = prepare_news_review_sample(
            cohort_path=args.cohort, max_selected=args.max_selected,
        )
        if (args.output.is_symlink() or args.output.exists()
                or not args.output.parent.is_dir()
                or args.output.parent.is_symlink()):
            raise ValueError("Output must be new file in existing ordinary directory")
        flags = os.O_WRONLY | os.O_EXCL | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(args.output, flags, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as out:
                json.dump(report, out, ensure_ascii=False, indent=2)
                out.write("\n")
        except BaseException:
            args.output.unlink(missing_ok=True)
            raise
    except (ValueError, TypeError, OSError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, f"news-review-sample: {type(exc).__name__}: {exc}\n")
    print(json.dumps({
        "status": report["status"],
        "totalArticles": report["totalArticles"],
        "candidateUnitCount": report["candidateUnitCount"],
        "multiArticleCandidateUnits": report["multiArticleCandidateUnits"],
        "crossUnitLexicalCandidatePairs": report["crossUnitLexicalCandidatePairs"],
        "selectedArticles": report["selectedArticles"],
        "observedDayCountsUTC": report["observedDayCountsUTC"],
        "trainingDatasetEligible": False,
        "humanGoldCreated": False,
        "stockPricesUsed": False,
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
