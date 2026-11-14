"""Read-only KR/US minute-price source compatibility report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tris_market_event_intelligence.market_minute_source import (
    MarketSourceContractError,
    audit_market_source,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit pinned KR/US minute-source CSV files")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    try:
        report = audit_market_source(args.manifest)
        if args.output is None:
            print(json.dumps(report, ensure_ascii=False))
        else:
            with args.output.open("x", encoding="utf-8") as target:
                json.dump(report, target, ensure_ascii=False, indent=2)
                target.write("\n")
            print(json.dumps({
                "status": "audit_only", "report": str(args.output),
                "markets": report["markets_inspected"],
            }))
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.exit(2, f"market-minute-audit: {error}\n")


if __name__ == "__main__":
    main()
