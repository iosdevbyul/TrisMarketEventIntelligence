"""Read-only source chronology and growth audit. No human gold required."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tris_market_event_intelligence.news_observation_coverage import (
    audit_news_observation_coverage,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare NEWS-only archived observations, not prediction quality"
    )
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--newer", type=Path)
    args = parser.parse_args()
    try:
        report = audit_news_observation_coverage(
            baseline_path=args.cohort, newer_path=args.newer,
        )
    except (ValueError, TypeError, OSError, KeyError,
            json.JSONDecodeError) as exc:
        parser.exit(2, f"news-observation-coverage: {type(exc).__name__}: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
