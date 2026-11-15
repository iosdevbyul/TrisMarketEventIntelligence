"""Create private offline NEWS review HTML; never start a server."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tris_market_event_intelligence.news_offline_review_page import (
    render_offline_review,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one private offline HTML human NEWS review page"
    )
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        rendered = render_offline_review(
            cohort_path=args.cohort, sample_path=args.sample,
        )
        if (args.output.suffix.lower() != ".html"
                or args.output.exists() or args.output.is_symlink()
                or not args.output.parent.is_dir()
                or args.output.parent.is_symlink()):
            raise ValueError("New private .html file in ordinary directory required")
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(args.output, flags, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as out:
                out.write(rendered)
        except BaseException:
            args.output.unlink(missing_ok=True)
            raise
    except (ValueError, TypeError, KeyError, OSError,
            json.JSONDecodeError) as exc:
        parser.exit(2, f"news-offline-review: {type(exc).__name__}: {exc}\n")
    print(json.dumps({
        "status": "PRIVATE_OFFLINE_REVIEW_PAGE_CREATED",
        "serverStarted": False,
        "humanGoldCreated": False,
        "modelCalls": 0,
        "networkCalls": 0,
        "stockPricesUsed": False,
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
