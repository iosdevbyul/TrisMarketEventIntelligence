"""General news + economic indicator evidence-to-market-pressure research.

Uses the existing validated ImpactResult, not regulator-specific state labels.
This is conditional NEWS PRESSURE, never a stock price/return prediction.
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, time
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tris_market_event_intelligence.impact import (
    SYSTEM_PROMPT,
    ImpactAnalysisError,
    ImpactResult,
)
from tris_market_event_intelligence.impact_mechanism import mechanism_quality_code
from tris_market_event_intelligence.impact_quality import rationale_quality_code
from tris_market_event_intelligence.llm_runtime import LLMProvider

PROMPT_VERSION = "multi-evidence-market-pressure-v1"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MarketEvidence(Strict):
    sourceId: str = Field(min_length=1)
    kind: Literal["NEWS", "ECONOMIC_INDICATOR"]
    publishedDate: date
    publishedAt: datetime | None = None
    sourceName: str = Field(min_length=1)
    sourceUrl: str
    excerpt: str = Field(min_length=20, max_length=1400)
    verification: Literal["VERIFIED_SOURCE", "REPORTED_UNVERIFIED"]
    indicatorName: str | None = None
    indicatorValue: str | None = None
    indicatorPrevious: str | None = None
    indicatorPeriod: str | None = None
    expectation: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> MarketEvidence:
        if self.publishedAt is not None:
            if (self.publishedAt.tzinfo is None
                    or self.publishedAt.utcoffset() is None):
                raise ValueError("MARKET_SOURCE_TIMEZONE_REQUIRED")
            if self.publishedAt.astimezone(UTC).date() != self.publishedDate:
                raise ValueError("MARKET_SOURCE_UTC_DATE_MISMATCH")
        url = urlsplit(self.sourceUrl)
        if (
            url.scheme != "https" or not url.hostname
            or url.username or url.password or url.fragment
            or url.port not in (None, 443)
        ):
            raise ValueError("MARKET_EVIDENCE_SOURCE_URL_INVALID")
        if self.kind == "ECONOMIC_INDICATOR" and not self.indicatorName:
            raise ValueError("ECONOMIC_INDICATOR_NAME_REQUIRED")
        if self.expectation and self.expectation not in self.excerpt:
            raise ValueError("UNSUPPORTED_EXPECTATION")
        if self.indicatorValue and self.indicatorValue not in self.excerpt:
            raise ValueError("UNSUPPORTED_INDICATOR_VALUE")
        if self.indicatorPrevious and self.indicatorPrevious not in self.excerpt:
            raise ValueError("UNSUPPORTED_PREVIOUS_VALUE")
        return self


class MarketEvidencePacket(Strict):
    eventId: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    asOfDateExclusive: date
    asOfTimestampExclusive: datetime | None = None
    evidences: list[MarketEvidence] = Field(min_length=1, max_length=24)

    @model_validator(mode="after")
    def check_ids(self) -> MarketEvidencePacket:
        if self.asOfTimestampExclusive is not None:
            cutoff = self.asOfTimestampExclusive
            if cutoff.tzinfo is None or cutoff.utcoffset() is None:
                raise ValueError("MARKET_CUTOFF_TIMEZONE_REQUIRED")
            if cutoff.astimezone(UTC).date() != self.asOfDateExclusive:
                raise ValueError("MARKET_CUTOFF_UTC_DATE_MISMATCH")
            if any(source.publishedAt is None for source in self.evidences):
                raise ValueError("PRECISE_REPLAY_REQUIRES_SOURCE_TIMESTAMPS")
        ids = [source.sourceId for source in self.evidences]
        if len(ids) != len(set(ids)):
            raise ValueError("MARKET_EVIDENCE_SOURCE_IDS_DUPLICATED")
        return self


def _digest(obj: dict) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def cutoff_visible(packet: MarketEvidencePacket, source: MarketEvidence) -> bool:
    """Timestamp-level cutoff with explicit UTC date-only legacy semantics."""
    cutoff = packet.asOfTimestampExclusive or datetime.combine(
        packet.asOfDateExclusive, time.min, tzinfo=UTC,
    )
    if source.publishedAt is not None:
        return source.publishedAt.astimezone(UTC) < cutoff.astimezone(UTC)
    # Date-only sources cannot enter a precise cutoff.
    if packet.asOfTimestampExclusive is not None:
        return False
    return source.publishedDate < packet.asOfDateExclusive


def _source_packet(packet: MarketEvidencePacket) -> dict:
    return {
        "eventId": packet.eventId,
        "asOfDateExclusive": packet.asOfDateExclusive.isoformat(),
        "asOfTimestampExclusive": (
            packet.asOfTimestampExclusive.isoformat()
            if packet.asOfTimestampExclusive is not None else None
        ),
        "evidences": [
            item.model_dump(mode="json")
            for item in packet.evidences
            if cutoff_visible(packet, item)
        ],
    }


def analyze_market_evidence(
    packet: MarketEvidencePacket, provider: LLMProvider,
) -> dict:
    """Generate one bounded multi-source analysis of conditional pressure."""
    visible = [
        item for item in packet.evidences
        if cutoff_visible(packet, item)
    ]
    if not visible:
        raise ValueError("MARKET_EVIDENCE_EMPTY_AT_CUTOFF")
    reported = any(s.verification != "VERIFIED_SOURCE" for s in visible)
    payload = _source_packet(packet)
    schema = ImpactResult.model_json_schema()
    schema["$defs"]["Impact"]["properties"]["uncertainty"]["enum"] = (
        ["HIGH"] if reported else ["LOW", "MEDIUM", "HIGH"]
    )
    raw = provider.generate_json(
        system=SYSTEM_PROMPT + """
Analyze the WHOLE supplied dated event evidence packet. It may contain
company news, industry developments and/or economic indicators.
First distinguish the observation from a predicted market effect.
For indicators, only infer a surprise relative to expectations when an
expectation is explicitly sourced. Never assume higher economic values
are always positive: explain discount rate, demand, earnings or cost
transmission separately for each affected market/sector.
A single incident can pressure different markets or sectors in
opposite directions. Cite verbatim excerpt text from a visible source.
Do not fabricate missing analyst expectations, base periods or quantities.
Never use post-cutoff evidence, market-price outcomes or trading signals.
The model's qualitative pressure is provisional, not verified returns.
""" + (
            "\nAt least one input is not independently verified. "
            "Every impact MUST have HIGH uncertainty."
            if reported else ""
        ),
        user=json.dumps({
            "eventId": packet.eventId,
            "subject": packet.subject,
            "asOfDateExclusive": packet.asOfDateExclusive.isoformat(),
            "evidence": payload["evidences"],
            "stockPricesAvailable": False,
            "outcomesAvailable": False,
        }, ensure_ascii=False),
        schema=schema,
    )
    result = ImpactResult.model_validate_json(raw)
    accepted = []
    for impact in result.impacts:
        if reported and impact.uncertainty != "HIGH":
            raise ImpactAnalysisError(
                "UNCERTAINTY_MISMATCH", "Unverified sources require HIGH uncertainty"
            )
        if rationale_quality_code(impact.rationale):
            raise ImpactAnalysisError(
                "RATIONALE_INCOMPLETE", "Impact rationale is incomplete"
            )
        if mechanism_quality_code(impact.sector, impact.rationale):
            raise ImpactAnalysisError(
                "MARKET_TRANSMISSION_MISSING", "Sector impact lacks mechanism"
            )
        matching = [
            item for item in visible
            if impact.evidence_excerpt in item.excerpt
        ]
        if not matching:
            raise ImpactAnalysisError(
                "EVIDENCE_MISMATCH", "Impact cites unavailable or post-cutoff evidence"
            )
        if any(not any(entity in src.excerpt for src in visible)
               for entity in impact.entities):
            raise ImpactAnalysisError(
                "ENTITY_MISMATCH", "Impact mentions unsourced entities"
            )
        accepted.append({
            **impact.model_dump(),
            "supportingSourceIds": sorted({source.sourceId for source in matching}),
            "observedMarketReaction": False,
            "pressureNotPriceForecast": True,
        })
    return {
        "schemaVersion": "market_evidence_pressure_research_v1",
        "eventId": packet.eventId,
        "asOfDateExclusive": packet.asOfDateExclusive.isoformat(),
        "modelId": provider.model_id,
        "promptVersion": PROMPT_VERSION,
        "visibleSourceIds": [s.sourceId for s in visible],
        "excludedFutureSourceIds": [
            s.sourceId for s in packet.evidences if not cutoff_visible(packet, s)
        ],
        "visibleEvidenceSha256": _digest(payload),
        "historicalPITVerified": False,
        "sourceFactsIndependentlyVerified": False,
        "semanticAccuracyVerified": False,
        "marketReturnsAnalyzed": False,
        "modelCalls": 1,
        "impacts": accepted,
        "status": "CONDITIONAL_MARKET_PRESSURE_REQUIRES_REVIEW",
    }
