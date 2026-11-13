"""Run the configured LLM impact provider against eligible events."""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from tris_market_event_intelligence.impact import OpenAICompatibleProvider
from tris_market_event_intelligence.impact_store import ImpactProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate research impact hypotheses")
    parser.add_argument(
        "--db", type=Path, default=Path(os.getenv("MARKET_EVENT_DB", "data/news.sqlite3"))
    )
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    api_key = os.getenv("MARKET_EVENT_LLM_API_KEY")
    model = os.getenv("MARKET_EVENT_LLM_MODEL")
    if not api_key or not model:
        parser.error("Set MARKET_EVENT_LLM_API_KEY and MARKET_EVENT_LLM_MODEL")
    provider = OpenAICompatibleProvider(
        base_url=os.getenv("MARKET_EVENT_LLM_BASE_URL", "https://api.openai.com/v1"),
        api_key=api_key,
        model=model,
    )
    stats = ImpactProcessor(args.db, provider).process_pending(args.batch_size)
    print(json.dumps(asdict(stats)))


if __name__ == "__main__":
    main()
