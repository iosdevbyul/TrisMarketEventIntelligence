"""Blinded same-news Gemma comparison with separately sealed model assignment.

Source evidence and candidate rationales are PRIVATE human review inputs.
No human-gold creation, stock-price observations, or automatic promotion.
"""
from __future__ import annotations

import hashlib
import json
import secrets
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from tris_market_event_intelligence.gemma4_news_comparison import (
    MODELS,
    _validate_models,
)
from tris_market_event_intelligence.impact import (
    PROMPT_VERSION,
    EventInput,
    ImpactAnalysisError,
    ImpactResult,
    validate_result,
)
from tris_market_event_intelligence.news_two_stage_pilot import select_news_events

PACKET_SCHEMA = "news_blind_review_packet_v1"
KEY_SCHEMA = "gemma4_blind_news_mapping_v1"
REVIEW_SCHEMA = "news_blind_human_judgments_v1"
MAX_NEWS = 3
RATINGS = {"SUPPORTED", "PARTIAL", "UNSUPPORTED", "UNASSESSABLE"}
VERDICTS = {"A", "B", "TIE", "BOTH_INADEQUATE", "UNDECIDED"}


class ReviewerModel(Protocol):
    model_id: str

    def analyze(self, event: EventInput) -> ImpactResult: ...


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def build_blind_packet(
    *, db_path: Path,
    analyzers: tuple[ReviewerModel, ReviewerModel],
    models: tuple[str, str] = MODELS,
    limit: int = 2, choose_first: Callable[[], bool] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Generate private blind packet and separate, never reviewer-shared key.

    Both models receive the identical original source and validator contract.
    Source excerpt is retained ONLY in the private reviewer packet.
    """
    _validate_models(models)
    if type(limit) is not int or not 1 <= limit <= MAX_NEWS:
        raise ValueError("Human review pilot must use 1..3 news events")
    if len(analyzers) != 2 or any(
        analyzer.model_id != name
        for analyzer, name in zip(analyzers, models, strict=True)
    ):
        raise ValueError("Model identities must match pinned local choices")
    events = select_news_events(db_path, limit=limit, model_id=models[0])
    items: list[dict[str, object]] = []
    assignments: list[dict[str, object]] = []
    flip = choose_first or (lambda: bool(secrets.randbits(1)))
    for event in events:
        news_hash = _hash({
            "eventId": event.event_id, "sourceUrl": event.source_url,
            "evidence": event.evidence, "status": event.status,
            "classification": event.classification,
        })
        results: list[dict[str, object]] = []
        for analyzer in analyzers:
            try:
                validated = validate_result(event, analyzer.analyze(event))
                results.append({
                    "status": "VALIDATED_OUTPUT",
                    "impacts": [
                        {
                            "market": impact.market, "sector": impact.sector,
                            "direction": impact.direction, "horizon": impact.horizon,
                            "rationale": impact.rationale,
                            "evidenceExcerpt": impact.evidence_excerpt,
                            "uncertainty": impact.uncertainty,
                            "drivers": impact.drivers, "entities": impact.entities,
                        } for impact in validated.impacts
                    ],
                })
            except ImpactAnalysisError as exc:
                results.append({
                    "status": "REJECTED_CONTRACT",
                    "failureCode": exc.code,
                    "impacts": [],
                })
            except Exception:  # noqa: BLE001
                results.append({
                    "status": "MODEL_RUNTIME_FAILED",
                    "failureCode": "MODEL_OR_NETWORK_ERROR",
                    "impacts": [],
                })
        reversed_order = flip()
        if type(reversed_order) is not bool:
            raise TypeError("Blind assignment must be a boolean")
        slots = (1, 0) if reversed_order else (0, 1)
        item = {
            "eventId": event.event_id,
            "newsInputSha256": news_hash,
            "sourceContext": {
                "classification": event.classification,
                "verificationStatus": event.status,
                "sourceUrl": event.source_url,
                "evidence": event.evidence,
                "claimKind": event.claim_kind,
            },
            "responses": {
                "A": results[slots[0]],
                "B": results[slots[1]],
            },
            "humanReview": {
                "newsAttribution": None,
                "A": {
                    "evidenceAccuracy": None, "contextAndMechanism": None,
                    "uncertaintyHonesty": None,
                },
                "B": {
                    "evidenceAccuracy": None, "contextAndMechanism": None,
                    "uncertaintyHonesty": None,
                },
                "preferredResponse": None,
                "reviewReason": None,
            },
        }
        assignments.append({
            "eventId": event.event_id,
            "newsInputSha256": news_hash,
            "A": models[slots[0]], "B": models[slots[1]],
        })
        items.append(item)
    packet = {
        "schemaVersion": PACKET_SCHEMA,
        "status": "REQUIRES_INDEPENDENT_HUMAN_REVIEW",
        "promptVersion": PROMPT_VERSION,
        "createdAt": datetime.now(UTC).isoformat(),
        "items": items,
        "modelIdentitiesDisclosed": False,
        "humanGoldAlreadyAvailable": False,
        "humanNewsQualityVerified": False,
        "promotionAllowed": False,
        "stockPricesUsed": False,
        "productionDatabaseModified": False,
    }
    key = {
        "schemaVersion": KEY_SCHEMA,
        "packetSha256": _hash(packet),
        "assignments": assignments,
        "revealOnlyAfterBlindReview": True,
        "promotionAllowed": False,
    }
    return packet, key


def _validate_judgment(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "newsAttribution", "A", "B", "preferredResponse", "reviewReason",
    }:
        raise ValueError("Reviewer must complete all required categories")
    if value["newsAttribution"] not in RATINGS:
        raise ValueError("Human news attribution grade required")
    for slot in ("A", "B"):
        row = value[slot]
        if not isinstance(row, dict) or set(row) != {
            "evidenceAccuracy", "contextAndMechanism", "uncertaintyHonesty",
        } or any(row[key] not in RATINGS for key in row):
            raise ValueError("Human source/context/uncertainty ratings required")
    if value["preferredResponse"] not in VERDICTS:
        raise ValueError("Valid blinded preference required")
    reason = value["reviewReason"]
    if not isinstance(reason, str) or not 20 <= len(reason.strip()) <= 1000:
        raise ValueError("Substantive independent human review reason required")
    return value


def audit_blind_judgments(
    packet: dict[str, object], reviewed: dict[str, object],
    *, reviewer: str,
) -> dict[str, object]:
    """Check human-entered grades against the original immutable blind packet.

    A finished review does not disclose the secret mapping and does not
    assert correctness of human grades or generalization on new news.
    """
    if not isinstance(reviewer, str) or not 2 <= len(reviewer.strip()) <= 100:
        raise ValueError("Independent reviewer identifier required")
    if not isinstance(packet, dict) or packet.get("schemaVersion") != PACKET_SCHEMA:
        raise ValueError("Original blind packet required")
    if not isinstance(reviewed, dict) or reviewed.get("schemaVersion") != REVIEW_SCHEMA:
        raise ValueError("Human review document required")
    if reviewed.get("humanReviewCompleted") is not True:\n        raise ValueError("Explicit human completion required")\n    if reviewed.get("packetSha256") != _hash(packet):
        raise ValueError("Human ratings refer to a different blind packet")
    decisions = reviewed.get("items")
    original = packet.get("items")
    if not isinstance(original, list) or not isinstance(decisions, list) or (
        not original or len(decisions) != len(original)
    ):
        raise ValueError("Human ratings must cover each original paired news item")
    counts: Counter[str] = Counter()
    seen: set[int] = set()
    for source, decision in zip(original, decisions, strict=True):
        if not isinstance(source, dict) or not isinstance(decision, dict):
            raise ValueError("Malformed original news or reviewed decision")
        article_id = source.get("eventId")
        if type(article_id) is not int or article_id in seen or (
            decision.get("eventId") != article_id
            or decision.get("newsInputSha256") != source.get("newsInputSha256")
        ):
            raise ValueError("Human review is misaligned with source evidence")
        seen.add(article_id)
        if set(decision) != {"eventId", "newsInputSha256", "humanReview"}:
            raise ValueError("Unexpected metadata in blinded human judgment")
        grade = _validate_judgment(decision["humanReview"])
        counts[grade["preferredResponse"]] += 1
    return {
        "schemaVersion": "gemma4_blind_human_review_audit_v1",
        "status": "HUMAN_JUDGMENTS_FORMAT_VERIFIED_NOT_GOLD_CERTIFIED",
        "packetSha256": _hash(packet),
        "reviewSha256": _hash(reviewed),
        "reviewer": reviewer.strip(),
        "pairedNewsReviewed": len(original),
        "blindPreferences": dict(sorted(counts.items())),
        "modelIdentitiesDisclosed": False,
        "independentHumanSourceVerificationProven": False,
        "humanJudgmentAccuracyCertified": False,
        "modelQualityImprovementValidated": False,
        "productionModelChanged": False,
        "promotionAllowed": False,
        "stockPricesUsed": False,
    }


def prepare_blind_judgments(packet: dict[str, object]) -> dict[str, object]:
    """Independent reviewer fills null values without seeing sealed mapping."""
    if not isinstance(packet, dict) or packet.get("schemaVersion") != PACKET_SCHEMA:
        raise ValueError("Original blind packet required")
    items = packet.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("No model responses available for blinded grading")
    return {
        "schemaVersion": REVIEW_SCHEMA,
        "packetSha256": _hash(packet),
        "items": [
            {
                "eventId": row["eventId"],
                "newsInputSha256": row["newsInputSha256"],
                "humanReview": json.loads(json.dumps(row["humanReview"])),
            } for row in items
        ],
        "humanReviewCompleted": False,
        "modelIdentitiesDisclosed": False,
        "productionModelChanged": False,
    }
