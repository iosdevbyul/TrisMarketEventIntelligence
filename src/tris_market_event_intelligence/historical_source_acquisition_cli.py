"""Explicit authorized historical price or macro vintage acquisition CLI."""
from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

import httpx

from tris_market_event_intelligence.historical_source_acquisition import (
    HistoricalProviderError,
    create_private_artifact,
    fred_revision_archive,
    json_bytes,
    tiingo_eod,
    verify_tiingo_export,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Opt-in single-source EOD or FRED history acquisition; no training"
    )
    mode = parser.add_subparsers(dest="source", required=True)
    price = mode.add_parser("tiingo")
    price.add_argument("--symbol", required=True)
    macro = mode.add_parser("fred")
    macro.add_argument("--series-id", required=True)
    for sub in (price, macro):
        sub.add_argument("--start", type=date.fromisoformat, required=True)
        sub.add_argument("--end", type=date.fromisoformat, required=True)
        sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        with httpx.Client(timeout=httpx.Timeout(35.0), trust_env=False) as session:
            if args.source == "tiingo":
                raw, metadata = tiingo_eod(
                    symbol=args.symbol, start=args.start, end=args.end,
                    token=os.getenv("TRIS_TIINGO_API_TOKEN", ""), client=session,
                )
                fingerprint = create_private_artifact(args.output, raw)
                sessions = verify_tiingo_export(args.output, fingerprint)
                print(json.dumps({
                    "status": "PRICE_HISTORY_ACQUIRED_UNVERIFIED",
                    "provider": "tiingo-eod", "symbol": args.symbol,
                    "firstDate": metadata["firstDate"], "lastDate": metadata["lastDate"],
                    "sessions": sessions, "sha256": fingerprint,
                    "independentProvenanceVerified": False,
                    "trainingReady": False,
                    "productionChanged": False,
                }))
            else:
                packet = fred_revision_archive(
                    series_id=args.series_id, start=args.start, end=args.end,
                    api_key=os.getenv("TRIS_FRED_API_KEY", ""), client=session,
                )
                digest = create_private_artifact(args.output, json_bytes(packet))
                print(json.dumps({
                    "status": "MACRO_DATE_ONLY_VINTAGES_QUARANTINED",
                    "seriesId": args.series_id, "rows": packet["rows"],
                    "sha256": digest,
                    "pointInTimeIntradayEligible": False,
                    "trainingReady": False,
                    "productionChanged": False,
                }))
    except (HistoricalProviderError, OSError) as exc:
        code = exc.args[0] if isinstance(exc, HistoricalProviderError) else "LOCAL_IO_ERROR"
        parser.exit(2, f"historical-source-acquisition: {code}\n")


if __name__ == "__main__":
    main()
