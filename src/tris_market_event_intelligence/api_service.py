"""Explicit path-bound, loopback-only API server for the opt-in macOS service."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from tris_market_event_intelligence.api import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve read-only claims on local loopback")
    base = Path.home() / "Library" / "Application Support" / "TrisMarketEventIntelligence"
    parser.add_argument("--db", type=Path, default=base / "news.sqlite3")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be 1..65535")
    db = args.db.expanduser().resolve()
    uvicorn.run(create_app(db), host="127.0.0.1", port=args.port, proxy_headers=False)


if __name__ == "__main__":
    main()
