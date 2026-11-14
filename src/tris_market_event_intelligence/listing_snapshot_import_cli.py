"""Explicit local review-only listing snapshot import; never fetch market APIs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tris_market_event_intelligence.listing_snapshot_import import export_snapshot_registry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate local hash-pinned KR/US listing snapshots, optional create-only CSV",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--write", action="store_true",
        help="Create a review-only registry CSV; never overwrite any existing file",
    )
    args = parser.parse_args()
    if args.write and args.output is None:
        parser.error("--write requires an explicit --output destination")
    if args.output is not None and not args.write:
        parser.error("--output requires --write")
    try:
        result = export_snapshot_registry(
            args.manifest.expanduser(),
            args.output.expanduser() if args.output else Path("."),
            write=args.write,
        )
    except (ValueError, TypeError, OSError, KeyError, UnicodeError) as exc:
        parser.exit(2, f"listing-snapshot: {type(exc).__name__}\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
