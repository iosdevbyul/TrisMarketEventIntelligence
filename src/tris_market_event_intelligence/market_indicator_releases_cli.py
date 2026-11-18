"""Explicit import of independently sourced macro releases into existing SQLite."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tris_market_event_intelligence.market_indicator_releases import (
    IndicatorReleaseBatch,
    IndicatorReleaseStore,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", action="store_true")
    mode.add_argument("--import", dest="import_rows", action="store_true")
    args = parser.parse_args()
    batch = IndicatorReleaseBatch.model_validate_json(
        args.input.read_text(encoding="utf-8")
    )
    if args.preview:
        print(json.dumps({
            "status": "INDICATOR_BATCH_VALIDATED_ONLY",
            "releases": len(batch.releases),
            "withConsensus": sum(x.consensus is not None for x in batch.releases),
            "modelCalls": 0, "dbWrites": 0,
            "historicalPITVerified": False,
        }))
        return
    store = IndicatorReleaseStore(args.db)
    store.initialize()
    result = store.import_batch(batch)
    print(json.dumps({
        "status": "RELEASES_IMPORTED_UNVERIFIED",
        **result, "modelCalls": 0,
        "historicalPITVerified": False,
    }))


if __name__ == "__main__":
    main()
