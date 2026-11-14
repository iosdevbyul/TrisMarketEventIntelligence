"""Offline, no-write readiness check for existing DonghakStockVision snapshots."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from tris_market_event_intelligence.krx_snapshot import (
    MarketSnapshotError,
    PinnedKRXSnapshot,
)


def inspect_snapshot(
    path: Path, sha256: str, tickers: tuple[str, ...],
) -> dict[str, object]:
    if not 1 <= len(tickers) <= 25 or len(set(tickers)) != len(tickers):
        raise ValueError("select 1..25 distinct tickers")
    snapshot = PinnedKRXSnapshot(path, expected_sha256=sha256)
    items = [snapshot.series(ticker) for ticker in tickers]
    return {
        "status": "ok",
        "source_kind": snapshot.kind,
        "source_sha256": sha256,
        "inspected_tickers": len(items),
        "price_eligible_tickers": sum(bool(item["event_study_price_eligible"]) for item in items),
        "benchmark_ready": False,
        "event_study_dataset_ready": False,
        "training_ready": False,
        "quality_verified": False,
        "note": (
            "Research-readiness audit only: raw stock close is NOT adjusted close "
            "and KRX stock-series DB does not provide an authenticated KOSPI index "
            "benchmark. Actual research requires independent benchmark, corporate-"
            "action adjusted price provenance and aligned event publication times."
        ),
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only KRX market snapshot readiness")
    parser.add_argument("--market-db", type=Path, required=True)
    parser.add_argument("--sha256", required=True, help="Pinned lowercase SHA-256 of offline DB")
    parser.add_argument("--tickers", required=True, help="Comma-separated 1..25 KRX tickers")
    args = parser.parse_args()
    try:
        selected = tuple(s.strip() for s in args.tickers.split(","))
        result = inspect_snapshot(args.market_db, args.sha256, selected)
    except (MarketSnapshotError, ValueError, OSError, sqlite3.Error) as exc:
        parser.exit(2, f"krx-readiness: {exc}\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
