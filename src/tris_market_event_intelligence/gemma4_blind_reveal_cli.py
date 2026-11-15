"""Explicit post-review reveal; private report only, no automated promotion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tris_market_event_intelligence.gemma4_blind_reveal import reveal_blind_review
from tris_market_event_intelligence.gemma4_blind_review_cli import (
    _create_private,
    _read_private,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reveal Gemma identities after complete human-blind NEWS reviews"
    )
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--review", type=Path, action="append", required=True)
    parser.add_argument("--reviewer", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if len(args.review) != len(args.reviewer):
            raise ValueError("Each completed review requires one reviewer identity")
        if args.output.resolve() in {
            args.packet.resolve(), args.mapping.resolve(),
            *(path.resolve() for path in args.review),
        }:
            raise ValueError("Revealed report must not replace any source file")
        # Mapping contents are handled only after each blind review passes audit.
        packet = _read_private(args.packet)
        reviews = [
            (reviewer, _read_private(path))
            for reviewer, path in zip(args.reviewer, args.review, strict=True)
        ]
        mapping = _read_private(args.mapping)
        report = reveal_blind_review(
            packet=packet, mapping=mapping, reviews=reviews,
        )
        _create_private(args.output, report)
    except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"gemma-blind-reveal: {type(exc).__name__}: {exc}\n")
    print(json.dumps({
        "status": report["status"],
        "reviewerCount": report["reviewerCount"],
        "pairedNewsReviewed": report["pairedNewsReviewed"],
        "report": str(args.output),
        "modelPromotionAllowed": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
