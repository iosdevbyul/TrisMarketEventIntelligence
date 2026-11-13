"""Local, explicit human review of a news event claim; no write HTTP endpoint."""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from tris_market_event_intelligence.claim_review import ClaimReviewManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Record an audited claim review")
    parser.add_argument("--db", type=Path, default=Path(
        os.getenv("MARKET_EVENT_DB", "data/news.sqlite3")
    ))
    parser.add_argument("--claim-id", type=int, required=True)
    parser.add_argument("--status", required=True, choices=(
        "UNDER_REVIEW", "CORROBORATED", "DISPUTED", "RETRACTED", "CONFIRMED",
    ))
    parser.add_argument("--evidence-article-id", type=int, required=True)
    parser.add_argument("--quote", required=True)
    parser.add_argument("--kind", required=True, choices=(
        "SUPPORTING_REPORT", "PRIMARY_DOCUMENT", "CORRECTION", "DENIAL",
    ))
    parser.add_argument("--reason", required=True)
    parser.add_argument("--expected-revision", type=int, required=True)
    parser.add_argument("--operation-id", help="Idempotency key for safe retries")
    args = parser.parse_args()
    try:
        result = ClaimReviewManager(args.db).record(
            claim_id=args.claim_id, status=args.status,
            evidence_article_id=args.evidence_article_id,
            evidence_quote=args.quote, evidence_kind=args.kind,
            reason=args.reason, expected_revision=args.expected_revision,
            operation_id=args.operation_id,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(asdict(result), ensure_ascii=False))


if __name__ == "__main__":
    main()
