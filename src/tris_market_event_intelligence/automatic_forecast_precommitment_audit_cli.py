"""Read-only local automatic forecast precommitment audit.

Local SHA chains and SQLite triggers are NOT external timestamps, signatures,
independently verified earliest public disclosure or prospective accuracy.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from tris_market_event_intelligence.automatic_forecast_precommitment import (
    inspect_commitments,
    verify_chain,
)


def audit(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise ValueError("Existing operator-owned SQLite file required")
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        if not db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='automatic_forecast_commitments'"
        ).fetchone():
            return {
                "status": "NOT_OPTED_IN", "commitments": 0,
                "trainingEligible": False, "prospectiveAccuracy": None,
            }
        return {
            "status": "COMPLETED",
            "commitmentCoverage": inspect_commitments(db),
            "localHashIntegrity": verify_chain(db),
            "firstPublicAvailabilityVerified": False,
            "externallyTimestamped": False,
            "prospectiveAccuracy": None,
            "trainingEligible": False,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(audit(args.db), ensure_ascii=False))
    except (ValueError, KeyError, sqlite3.Error, OSError) as exc:
        parser.exit(2, f"forecast-precommitment-audit: {exc}\n")


if __name__ == "__main__":
    main()
