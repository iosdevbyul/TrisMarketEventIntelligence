"""Explicit read-only whole-ledger integrity audit."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from tris_market_event_intelligence.ticker_review_ledger_audit import audit_review_ledger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit every post-hoc ticker review row without modifying any database",
    )
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit_review_ledger(args.ledger)
    except (OSError, ValueError, TypeError, sqlite3.Error):
        parser.exit(2, "ticker-ledger-audit: verification failed\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
