"""Replay previous private Gemma responses against current validators, offline.

Zero Ollama/network calls. Do not upload the raw input trace or interpret
validator acceptance as independently verified economic accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from tris_market_event_intelligence.market_evidence_pressure import (
    MarketEvidencePacket,
)
from tris_market_event_intelligence.market_pressure_reasoning import (
    analyze_pressure_two_stage,
)


class _RecordedResponses:
    def __init__(self, responses: list[str], model_id: str) -> None:
        self.responses = iter(responses)
        self.calls = 0
        self.model_id = model_id
        self.settings = SimpleNamespace(kind="ollama")

    def generate_json(
        self, *, system: str, user: str, schema: dict | None,
    ) -> str:
        del system, user, schema
        self.calls += 1
        try:
            return next(self.responses)
        except StopIteration as exc:
            raise ValueError("REPLAY_TRACE_RESPONSE_MISSING") from exc


def replay_trace(path: Path) -> dict:
    requests: dict[int, dict] = {}
    responses: dict[int, str] = {}
    fact_calls: list[int] = []
    impact_calls: list[int] = []
    abstention_calls: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        call = record.get("call")
        stage = record.get("stage")
        if record.get("event") == "LLM_REQUEST" and isinstance(call, int):
            requests[call] = record
            if stage == "FACT_EXTRACTION":
                fact_calls.append(call)
            elif stage == "IMPACT_GENERATION":
                impact_calls.append(call)
            elif stage == "ABSTENTION_REASON":
                abstention_calls.append(call)
        elif record.get("event") == "LLM_RESPONSE" and isinstance(call, int):
            responses[call] = record["rawResponse"]

    results = []
    for fact_call in fact_calls:
        next_fact = next(
            (call for call in fact_calls if call > fact_call),
            float("inf"),
        )
        impacts = [
            call for call in impact_calls if fact_call < call < next_fact
        ]
        if len(impacts) != 1:
            raise ValueError("REPLAY_IMPACT_STAGE_NOT_UNIQUE")
        impact_call = impacts[0]
        downstream = [
            call for call in abstention_calls if impact_call < call < next_fact
        ]
        calls = [fact_call, impact_call, *downstream]
        if any(call not in responses for call in calls):
            raise ValueError("REPLAY_MODEL_RESPONSE_MISSING")
        request = json.loads(requests[fact_call]["userInput"])
        packet = MarketEvidencePacket.model_validate({
            "eventId": request["eventId"],
            "subject": request["subject"],
            "asOfDateExclusive": request["asOfDateExclusive"],
            "asOfTimestampExclusive": request.get("asOfTimestampExclusive"),
            "evidences": request["visibleSources"],
        })
        provider = _RecordedResponses(
            [responses[call] for call in calls],
            requests[fact_call].get("modelId", "recorded-gemma"),
        )
        report = analyze_pressure_two_stage(
            packet, provider, recover_invalid_impacts=True,
        )
        results.append({
            "eventId": packet.eventId,
            "sourceIds": report["visibleSourceIds"],
            "status": report["status"],
            "originalProposalCount": report["proposedImpactCount"],
            "acceptedImpacts": [
                {
                    "proposalIndex": item["proposalIndex"],
                    "targetType": item["targetType"],
                    "targetName": item["targetName"],
                    "direction": item["direction"],
                    "strength": item["strength"],
                    "supportingSourceIds": item["supportingSourceIds"],
                    "reasoningQuality": item["reasoningQuality"],
                }
                for item in report["impacts"]
            ],
            "rejectedImpacts": report["rejectedImpacts"],
            "semanticAccuracyVerified": False,
        })
    return {
        "schemaVersion": "offline_market_pressure_replay_v1",
        "readOnly": True,
        "newModelCalls": 0,
        "dbWrites": 0,
        "events": results,
        "acceptedImpactCount": sum(
            len(group["acceptedImpacts"]) for group in results
        ),
        "rejectedImpactCount": sum(
            len(group["rejectedImpacts"]) for group in results
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay_trace(args.trace), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
