"""Private blind NEWS review CLI; never merges sealed model IDs into review."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import httpx

from tris_market_event_intelligence.gemma4_blind_review import (
    audit_blind_judgments,
    build_blind_packet,
    prepare_blind_judgments,
)
from tris_market_event_intelligence.gemma4_news_comparison_cli import (
    local_model_preflight,
)
from tris_market_event_intelligence.gemma4_news_comparison import _validate_models
from tris_market_event_intelligence.impact import OpenAICompatibleProvider

MAX_PRIVATE_BYTES = 250000


def _read_private(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_PRIVATE_BYTES:
        raise ValueError("Existing bounded ordinary private review file required")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise TypeError("Blind review JSON must be an object")
    return result


def _create_private(path: Path, document: dict[str, object]) -> None:
    if path.is_symlink() or path.exists() or not path.parent.is_dir():
        raise ValueError("Create-only private output must have an existing directory")
    raw = (json.dumps(
        document, sort_keys=True, ensure_ascii=False, indent=2,
    ) + "\n").encode("utf-8")
    if len(raw) > MAX_PRIVATE_BYTES:
        raise ValueError("Private review packet is too large")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare or audit model-blinded NEWS-only human review"
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--audit", action="store_true")
    root = Path.home() / "Library/Application Support/TrisMarketEventIntelligence"
    parser.add_argument("--db", type=Path, default=root / "news.sqlite3")
    parser.add_argument("--gemma4", default="gemma4:e4b")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--reviewer")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.run:
            models = ("gemma2:latest", args.gemma4)
            _validate_models(models)
            if args.packet is None or args.mapping is None:
                raise ValueError("--run requires --packet and --mapping new private files")
            if args.packet.resolve() == args.mapping.resolve():
                raise ValueError("Blind packet and sealed mapping must be separate files")
            if any(p.is_symlink() or p.exists() or not p.parent.is_dir()
                   for p in (args.packet, args.mapping)):
                raise ValueError("Both private output paths must be unused")
            with httpx.Client(timeout=5, trust_env=False) as client:
                preflight = local_model_preflight(models, client=client)
            if not preflight["eachRequestedModelInstalled"]:
                raise ValueError("Both exact model tags must be locally installed")
            analyzers = tuple(
                OpenAICompatibleProvider(
                    base_url="http://127.0.0.1:11434/v1",
                    api_key="local-placeholder", model=name,
                    native_ollama=True, timeout_seconds=300,
                ) for name in models
            )
            packet, mapping = build_blind_packet(
                db_path=args.db.expanduser(), models=models,
                analyzers=analyzers, limit=args.limit,
            )
            if not packet["items"]:
                raise ValueError("No eligible archived NEWS events to blind review")
            # Mapping is created first so no published blind packet can be
            # left without its corresponding assignment record.
            _create_private(args.mapping, mapping)
            try:
                _create_private(args.packet, packet)
            except BaseException:
                args.mapping.unlink(missing_ok=True)
                raise
            status = "BLIND_PACKET_AND_SEALED_MAPPING_CREATED"
        elif args.prepare:
            if args.packet is None or args.output is None:
                raise ValueError("--prepare requires --packet and --output")
            result = prepare_blind_judgments(_read_private(args.packet))
            _create_private(args.output, result)
            status = "UNFILLED_HUMAN_REVIEW_TEMPLATE_CREATED"
        else:
            if any(x is None for x in (args.packet, args.review, args.output, args.reviewer)):
                raise ValueError("--audit requires packet, review, reviewer and output")
            result = audit_blind_judgments(
                _read_private(args.packet), _read_private(args.review),
                reviewer=args.reviewer,
            )
            _create_private(args.output, result)
            status = result["status"]
        print(json.dumps({
            "status": status,
            "modelNamesShownToHumanReviewer": False,
            "modelPromotionAllowed": False,
            "modelDownloadedOrInstalled": False,
        }, ensure_ascii=False))
    except (ValueError, TypeError, OSError, KeyError, httpx.HTTPError,
            json.JSONDecodeError) as exc:
        parser.exit(2, f"gemma-blind-review: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
