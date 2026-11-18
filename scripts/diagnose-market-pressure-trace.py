#!/usr/bin/env python3
"""Read private Gemma trace offline to explain impact-stage guard rejections.

No Ollama, HTTP, SQLite writes, credentials or new model calls. Do not
upload raw JSONL traces. Output reports claim/entity labels, not articles.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def inspect_trace(path: Path) -> dict:
    requests: dict[int, dict] = {}
    results: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("stage") != "IMPACT_GENERATION":
            continue
        call = record.get("call")
        if record.get("event") == "LLM_REQUEST":
            requests[call] = json.loads(record["userInput"])
        if record.get("event") != "LLM_RESPONSE":
            continue
        original = json.loads(record["rawResponse"])
        claims = requests.get(call, {}).get("groundedClaims", [])
        impacts = original.get("impacts", [])
        if not isinstance(impacts, list):
            results.append({"call": call, "error": "IMPACTS_NOT_ARRAY"})
            continue
        diagnostics = []
        for idx, impact in enumerate(impacts):
            if not isinstance(impact, dict):
                diagnostics.append({
                    "index": idx, "error": "IMPACT_NOT_OBJECT",
                })
                continue
            indexes = impact.get("claimIndexes", [])
            refs = [
                claims[i] for i in indexes
                if isinstance(i, int) and not isinstance(i, bool)
                and 0 <= i < len(claims)
            ] if isinstance(indexes, list) else []
            excerpts = [x["excerpt"] for x in refs]
            entities = impact.get("entities", [])
            unmatched = [
                name for name in entities
                if not isinstance(name, str)
                or not any(name in excerpt for excerpt in excerpts)
            ] if isinstance(entities, list) else ["ENTITIES_NOT_ARRAY"]
            direction = impact.get("direction")
            strength = impact.get("strength")
            diagnostics.append({
                "index": idx,
                "direction": direction,
                "strength": strength,
                "targetType": impact.get("targetType"),
                "targetName": impact.get("targetName"),
                "claimIndexes": indexes,
                "validClaimIndexes": len(refs) == len(indexes)
                if isinstance(indexes, list) else False,
                "uncitedEntities": unmatched,
                "directionStrengthConflict": (
                    direction == "UNCERTAIN" and strength != "UNASSESSED"
                ),
                "evidenceExcerptMatchesCitedClaims": any(
                    impact.get("evidence_excerpt") in excerpt
                    for excerpt in excerpts
                ) if isinstance(impact.get("evidence_excerpt"), str) else False,
            })
        results.append({
            "call": call, "impactCount": len(impacts),
            "claimCount": len(claims),
            "impacts": diagnostics,
        })
    return {
        "schemaVersion": "private_market_impact_trace_diagnostics_v1",
        "readOnly": True, "modelCalls": 0,
        "impactCalls": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_trace(args.trace), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
