"""Retrospective multi-hypothesis research, with optional human-reviewed learning notes.

In-context coaching and reviewed examples DO NOT train Gemma's weights.
The sealed forward prediction protocol is separate and remains untouched.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from tris_market_event_intelligence.historical_story_review import (
    CATEGORIES,
    prepare_historical_story_review,
)
from tris_market_event_intelligence.llm_runtime import LLMProvider

VERSION = "historical-multi-evidence-v1"
MAX_INPUT = 40_000
MAX_OUTPUT = 24_000
MAX_MEMORY = 20_000
MAX_REVIEW_NOTES = 6
MAX_STORY_ROWS = 100
PRIMARY_ID = "issuer-primary"


def _digest(doc: dict) -> str:
    return hashlib.sha256(json.dumps(
        doc, ensure_ascii=False, sort_keys=True,
    ).encode()).hexdigest()


def read_reviewed_lessons(path: Path | None) -> tuple[list[dict], str | None]:
    """Load only explicit operator-reviewed principles, never a model self-grade."""
    if path is None:
        return [], None
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_MEMORY:
        raise ValueError("Invalid bounded human-reviewed lessons file")
    raw = path.read_bytes()
    doc = json.loads(raw)
    if (not isinstance(doc, dict)
            or doc.get("schemaVersion") != "human_reviewed_research_lessons_v1"
            or doc.get("reviewedByHuman") is not True
            or not isinstance(doc.get("lessons"), list)
            or not 1 <= len(doc["lessons"]) <= MAX_REVIEW_NOTES):
        raise ValueError("Only human-reviewed lessons are eligible")
    lessons = []
    for entry in doc["lessons"]:
        if (not isinstance(entry, dict) or set(entry) != {"lesson", "evidence"}
                or any(not isinstance(entry[key], str)
                       or not 10 <= len(entry[key]) <= 300
                       for key in ("lesson", "evidence"))):
            raise ValueError("Invalid reviewed research lesson")
        lessons.append(dict(entry))
    return lessons, hashlib.sha256(raw).hexdigest()


def _prepared(story: dict) -> dict:
    if (story.get("schemaVersion") != "historical_story_review_v1"
            or story.get("status") != "URL_METADATA_ONLY_REVIEW_REQUIRED"
            or story.get("retrospectiveOutcomeKnown") is not True
            or story.get("sourcePermissionsVerified") is not False
            or story.get("forwardPredictionSealed") is not False):
        raise ValueError("Only unverified retrospective story packets supported")
    rows = story.get("references")
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_STORY_ROWS:
        raise ValueError("Bounded reference cohort required")
    if len({row["referenceId"] for row in rows}) != len(rows):
        raise ValueError("Duplicate evidence reference IDs")
    return story


def build_research_context(
    *, story: dict, lessons: list[dict] | None = None,
) -> dict:
    """Present ALL archived URL hints with primary source as separate evidence."""
    story = _prepared(story)
    lessons = lessons or []
    if len(lessons) > MAX_REVIEW_NOTES:
        raise ValueError("Reviewed memory exceeds bound")
    by_category: dict[str, list[dict]] = {category: [] for category in CATEGORIES}
    for row in story["references"]:
        category = row["candidateCategory"]
        if category not in by_category:
            raise ValueError("Unknown URL heuristic category")
        by_category[category].append({
            "referenceId": row["referenceId"],
            "urlDerivedHint": row["urlDerivedHint"][:160],
            "sourceHost": row["sourceHost"][:100],
            "archiveSlotUtc": row["gkgBatchSlotUtc"],
            "publicationTimestampVerified": False,
            "articleContentRead": False,
        })
    primary = story["officialPrimarySource"]
    if primary is not None:
        if primary.get("releaseClockIndependentlyVerified") is not False:
            raise ValueError("Primary clock verification contract changed")
        primary = {
            "referenceId": PRIMARY_ID, "name": primary["name"],
            "url": primary["url"], "date": primary["date"],
            "reportedRevenueUsdBillions": primary["reportedRevenueUsdBillions"],
            "outlookNextQuarterUsdBillions": primary["outlookNextQuarterUsdBillions"],
            "firstTradableAvailabilityVerified": False,
        }
    return {
        "task": "retrospective_explanation_only",
        "ticker": story["ticker"],
        "shockDate": story["shockDate"],
        "outcomeMaskedFromModel": True,
        "allReferencesProvided": len(story["references"]),
        "referencesByHeuristicCategory": by_category,
        "firstPartyDisclosureSeparateFromUrlHints": primary,
        "candidateStoryGroupsUnverified": story["candidateStoryGroups"],
        "economicIndicatorsActuallyAvailable": False,
        "humanReviewedMethodLessons": lessons,
    }


SYSTEM = """You are Gemma, a research assistant for an INDEPENDENT market-event
intelligence project. You do NOT change your own model weights by responding.
Goal: learn a disciplined research METHOD across historical cases, not
memorize future outcomes or assert a historical event caused returns.
This is BACKWARD EXPLANATION, never a sealed predictive test.

METHOD:
(1) Read ALL provided candidate references grouped by heuristic topic.
(2) Treat issuer earnings/guidance and dated first-party disclosures as
    a separate stronger *documentary* class than a secondary URL hint;
    do not blindly prioritize first-party evidence if unrelated.
(3) An issuer's sales outlook and demand signals may change expectations.
    Distinguish economic mechanism from confirmed causal price attribution.
(4) Compare at least two competing explanations when available:
    firm-specific, industry/competitor, macro/market, and other context.
(5) Separate URL-derived guesses, original publisher timestamp (UNKNOWN),
    approximate GKG ingestion batch, and corroborated issuer facts.
(6) Regional reposts and matching URL words do NOT prove independent events.
(7) Do NOT claim full articles read, actual publication clocks known, or
    which day-to-day return was caused by each candidate.
(8) Mention missing macro indicators explicitly, do NOT fabricate releases.
(9) Cite ONLY the given reference IDs, using 'issuer-primary' when provided;
    refer to it as a primary document, not a timestamp-verified trading signal.
(10) If evidence is insufficient, select UNASSESSED; no automatic promotion.
(11) Evaluate suggestions against alternatives, mention unavailable data.
(12) Human-reviewed method lessons, if present, guide discipline but are
     not truth labels or permission to use future outcomes.

GOOD REASONING EXAMPLE (hypothetical): a dated issuer statement can
support a hypothesis of revised expected revenue; competition headlines
might be background; a GKG batch timestamp cannot prove first publication.
BAD EXAMPLE: count four reprints as four market shocks or infer that
AMD GPU coverage definitely caused NVDA's observed return.

Return a valid JSON object with three *distinct candidate hypotheses*:
one plausible firm-specific explanation when supported, a substantive
alternative or confounder, and an uncertainty/missing-evidence candidate.
If fewer candidates have any support, use UNASSESSED instead of fabrication.
The top hypothesis must be selected by independent source strength and
economic mechanism, NOT by repeating stories or by seeing the outcome.
"""


SCHEMA: dict[str, object] = {
    "type": "object", "additionalProperties": False,
    "required": ["hypotheses", "research_summary", "missing_evidence",
                 "causal_proof", "prospective_prediction"],
    "properties": {
        "hypotheses": {
            "type": "array", "minItems": 3, "maxItems": 3,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["category", "reference_ids", "mechanism",
                             "alternative_explanations", "source_limitations",
                             "evidence_strength"],
                "properties": {
                    "category": {"type": "string", "enum": [*CATEGORIES, "UNASSESSED"]},
                    "reference_ids": {"type": "array", "items": {"type": "string"}},
                    "mechanism": {"type": "string"},
                    "alternative_explanations": {"type": "string"},
                    "source_limitations": {"type": "string"},
                    "evidence_strength": {"type": "string",
                                          "enum": ["PRIMARY_DOCUMENT",
                                                   "URL_HINT_ONLY", "INSUFFICIENT"]},
                },
            },
        },
        "research_summary": {"type": "string"},
        "missing_evidence": {"type": "string"},
        "causal_proof": {"type": "boolean", "enum": [False]},
        "prospective_prediction": {"type": "boolean", "enum": [False]},
    },
}


def compare_historical_hypotheses(
    *, story: dict, provider: LLMProvider,
    lessons: list[dict] | None = None, lesson_sha: str | None = None,
) -> dict:
    context = build_research_context(story=story, lessons=lessons)
    input_json = json.dumps(context, ensure_ascii=False)
    if len(input_json.encode()) > MAX_INPUT:
        raise ValueError("Bounded whole-cohort input exceeded resource cap")
    response = provider.generate_json(
        system=SYSTEM, user=input_json, schema=SCHEMA,
    )
    if not isinstance(response, str) or len(response.encode()) > MAX_OUTPUT:
        raise ValueError("Oversized model response")
    data = json.loads(response)
    if not isinstance(data, dict) or set(data) != set(SCHEMA["required"]):
        raise ValueError("Research model contract violated")
    if data["causal_proof"] is not False or data["prospective_prediction"] is not False:
        raise ValueError("Model cannot certify causality or forecast skill")
    rows = data["hypotheses"]
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("Require three competing hypotheses")
    allowed = {r["referenceId"] for r in story["references"]}
    if story["officialPrimarySource"] is not None:
        allowed.add(PRIMARY_ID)
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(
            SCHEMA["properties"]["hypotheses"]["items"]["required"]
        ):
            raise ValueError("Malformed candidate hypothesis")
        references = row["reference_ids"]
        if (row["category"] not in (*CATEGORIES, "UNASSESSED")
                or row["evidence_strength"] not in (
                    "PRIMARY_DOCUMENT", "URL_HINT_ONLY", "INSUFFICIENT"
                ) or not isinstance(references, list)
                or len(references) > 12
                or any(not isinstance(ref, str) or ref not in allowed
                       for ref in references)
                or len(references) != len(set(references))):
            raise ValueError("Unsupported source reference or evidence class")
        if (row["evidence_strength"] == "PRIMARY_DOCUMENT"
                and PRIMARY_ID not in references):
            raise ValueError("Primary evidence not cited")
        for key in ("mechanism", "alternative_explanations", "source_limitations"):
            if not isinstance(row[key], str) or not 10 <= len(row[key]) <= 1200:
                raise ValueError("Missing substantive research caveats")
    if (not isinstance(data["research_summary"], str)
            or not 10 <= len(data["research_summary"]) <= 1500
            or not isinstance(data["missing_evidence"], str)
            or not 10 <= len(data["missing_evidence"]) <= 1500):
        raise ValueError("Model omitted research limitation")
    return {
        "schemaVersion": "historical_multi_hypothesis_research_v1",
        "status": "RETROSPECTIVE_HYPOTHESES_UNVERIFIED",
        "modelId": provider.model_id, "promptVersion": VERSION,
        "reviewPacketSha256": _digest(story),
        "humanReviewedLessonsSha256": lesson_sha,
        "usedHumanReviewedLessons": bool(lessons),
        "allCandidateReferenceCount": len(story["references"]),
        "inputCategoryCounts": dict(Counter(
            r["candidateCategory"] for r in story["references"]
        )),
        "primarySourcePresented": story["officialPrimarySource"] is not None,
        "hypotheses": rows,
        "researchSummary": data["research_summary"],
        "missingEvidence": data["missing_evidence"],
        "modelCalls": 1,
        "noArticleTextsOrMacroProvided": True,
        "prospectivePrediction": False,
        "independentGoldCreated": False,
        "causalAttributionVerified": False,
        "forecastImprovementMeasured": False,
        "weightTrainingPerformed": False,
        "productionDbModified": False,
        "promotionAllowed": False,
    }


def research_from_scout(
    path: Path, *, provider: LLMProvider,
    reviewed_lessons: Path | None = None,
) -> dict:
    story = prepare_historical_story_review(path)
    lessons, sha = read_reviewed_lessons(reviewed_lessons)
    return compare_historical_hypotheses(
        story=story, provider=provider, lessons=lessons, lesson_sha=sha,
    )
