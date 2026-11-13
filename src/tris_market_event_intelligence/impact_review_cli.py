"""Read-only offline review export/scoring of stored impact hypotheses."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tris_market_event_intelligence.impact_review import (
    export_impact_review,
    score_impact_review,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline human impact quality audit")
    base = Path.home() / "Library" / "Application Support" / "TrisMarketEventIntelligence"
    parser.add_argument("--db", type=Path, default=Path(
        os.getenv("MARKET_EVENT_DB", str(base / "news.sqlite3"))
    ))
    sub = parser.add_subparsers(dest="action", required=True)
    export = sub.add_parser("export")
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--count", type=int, default=30)
    export.add_argument("--model-id")
    score = sub.add_parser("score")
    score.add_argument("--sheet", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.action == "export":
            result = export_impact_review(
                args.db.expanduser(), args.output.expanduser(),
                count=args.count, model_id=args.model_id,
            )
        else:
            result = score_impact_review(args.db.expanduser(), args.sheet.expanduser())
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
