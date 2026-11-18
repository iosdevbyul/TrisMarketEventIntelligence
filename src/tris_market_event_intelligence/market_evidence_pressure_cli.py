"""Analyze a user-supplied, cutoff-scoped news and economic-indicator packet."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from tris_market_event_intelligence.llm_runtime import (
    LLMCapabilities,
    LLMSettings,
)
from tris_market_event_intelligence.market_evidence_db_bridge import make_packet_from_db
from tris_market_event_intelligence.market_evidence_pressure import (
    MarketEvidencePacket,
    analyze_market_evidence,
)
from tris_market_event_intelligence.news_llm_factory import build_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--packet", type=Path)
    source.add_argument("--db", type=Path)
    parser.add_argument("--article-id", action="append", type=int, default=[])
    parser.add_argument("--economic-event-id", action="append", type=int, default=[])
    parser.add_argument("--event-id", type=str)
    parser.add_argument("--subject", type=str)
    parser.add_argument("--as-of", type=str)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default="gemma4:e2b")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.db:
        if not all((args.event_id, args.subject, args.as_of)):
            parser.error("--db requires --event-id, --subject and --as-of")
        packet = make_packet_from_db(
            args.db, event_id=args.event_id, subject=args.subject,
            as_of=args.as_of,
            article_ids=tuple(args.article_id),
            economic_event_ids=tuple(args.economic_event_id),
        )
    else:
        if args.article_id or args.economic_event_id or any(
            (args.event_id, args.subject, args.as_of)
        ):
            parser.error("--packet cannot be combined with DB selection flags")
        packet = MarketEvidencePacket.model_validate_json(
            args.packet.read_text(encoding="utf-8")
        )
    visible = [
        s for s in packet.evidences
        if s.publishedDate < packet.asOfDateExclusive
    ]
    if args.preview:
        print(json.dumps({
            "status": "CONDITIONAL_MARKET_EVIDENCE_PREVIEW",
            "eventId": packet.eventId,
            "visibleSourceIds": [s.sourceId for s in visible],
            "excludedFutureSourceIds": [
                s.sourceId for s in packet.evidences
                if s.publishedDate >= packet.asOfDateExclusive
            ],
            "modelCalls": 0,
            "historicalPITVerified": False,
        }))
        return
    if not visible:
        parser.error("MARKET_EVIDENCE_EMPTY_AT_CUTOFF")
    if args.output is None:
        parser.error("--run requires --output")
    if args.output.exists() or args.output.is_symlink():
        parser.error("OUTPUT_ALREADY_EXISTS")
    cfg = replace(
        LLMSettings.from_environment(explicit=True),
        model_id=args.model,
        capabilities=LLMCapabilities(
            json_schema=True, thinking=False, context_window=8192,
            max_output_tokens=4096,
        ),
    )
    runtime = build_runtime(cfg, confirm_model=True)
    result = analyze_market_evidence(packet, runtime)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps({
        "status": result["status"],
        "visibleSources": len(result["visibleSourceIds"]),
        "impactHypotheses": len(result["impacts"]),
        "modelCalls": result["modelCalls"],
        "historicalPITVerified": False,
        "outputCreated": True,
    }))


if __name__ == "__main__":
    main()
