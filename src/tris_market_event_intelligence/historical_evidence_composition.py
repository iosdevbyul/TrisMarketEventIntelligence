"""Evidence-first, source-scoped model narration with independent question axes.

Only cutoff-visible official claims and validated model-authored phase-one
changes enter the model. No retrospective question transitions or price data.
"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tris_market_event_intelligence.historical_evidence_scope import (
    is_observed_market_result,
)
from tris_market_event_intelligence.llm_runtime import LLMProvider


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceExplanation(Strict):
    sourceId: str
    claimId: str
    explanation: str = Field(min_length=35, max_length=310)
    limitation: str = Field(min_length=25, max_length=230)


class QuestionAxes(Strict):
    questionId: str
    regulatoryActionProgress: Literal[
        "NOT_OBSERVED", "ANNOUNCED", "AUTHORIZED", "IMPLEMENTING",
        "COMPLETED", "UNKNOWN",
    ]
    requestedEvidenceStatus: Literal[
        "NOT_OBSERVED", "PARTIAL", "DIRECTLY_OBSERVED",
    ]
    sourceId: str
    claimId: str
    explanation: str = Field(min_length=35, max_length=280)
    remainingEvidence: str = Field(min_length=25, max_length=280)


class AllQuestionAxes(Strict):
    assessments: list[QuestionAxes] = Field(default_factory=list, max_length=12)


_SOURCE_SYSTEM = """Explain ONE dated official source with a short complete factual
sentence, using only the given source claims and already validated model
change assertions. Give the exact sourceId and a claimId from the source.
Name action type, regulator and relevant product/aircraft scope. Distinguish
recommendation, authorization, implementation and completion. Use no
unseen facts, prices, inferred causality, invented outcomes or future data.
Do not repeat opaque IDs in the explanation. Identify specific evidence
that the source does not establish in limitation. End each field with a full
sentence and punctuation. Return strictly schema-conformant JSON."""

_QUESTIONS_SYSTEM = """Assess each research question ONLY against the
provided date-cutoff-visible official sources, not future transitions.
Give exactly one row per question. The axes are INDEPENDENT:
regulatoryActionProgress is the stage explicitly supported by a relevant
visible official claim (NOT_OBSERVED, ANNOUNCED, AUTHORIZED,
IMPLEMENTING, COMPLETED, UNKNOWN). Approval of a PROCEDURE does not
prove execution of individual inspections or completion of an audit.
requestedEvidenceStatus is about the exact DOCUMENT or answer requested:
NOT_OBSERVED (missing), PARTIAL (some but not all required direct evidence)
or DIRECTLY_OBSERVED (the exact requested answer exists in a visible
official source). A press release can directly answer an authorization
question without proving underlying review records. Cite an exact visible
sourceId and claimId; for NOT_OBSERVED, cite the visible basis of the
question, never invent a source. Explain precisely what is and is not
observed. Keep aircraft/product scope distinct. Do not use curator future
outcomes, prices, financial causality or later history. Strict JSON only."""


class ComposedEvidenceError(ValueError):
    """Fail closed when source or question evidence cannot be narrated."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _complete(text: str) -> bool:
    return text.rstrip().endswith((".", "!", "?"))


def _refs(packet: dict) -> set[tuple[str, str]]:
    return {
        (source["sourceId"], claim["claimId"])
        for source in packet["eligibleSources"] for claim in source["claims"]
    }


def generate_composed_explanation(
    provider: LLMProvider, packet: dict, draft: dict,
) -> dict:
    """Generate and validate one explanation per new source, then compose.

    The final prose is deterministic concatenation of source-scoped model
    sentences; model text is preserved verbatim and no facts are fabricated.
    """
    previous = set(packet["previouslySeenSourceIds"])
    new_sources = [
        source for source in packet["eligibleSources"]
        if source["sourceId"] not in previous
    ]
    if not new_sources:
        raise ComposedEvidenceError("COMPOSED_NO_NEW_SOURCE")
    available = _refs(packet)
    changes = draft["modelAuthoredChangeItems"]
    explanations = []
    diagnostics = []
    for source in new_sources:
        source_id = source["sourceId"]
        grounded = [x for x in changes if x["sourceId"] == source_id]
        if not grounded:
            raise ComposedEvidenceError("COMPOSED_SOURCE_CHANGE_MISSING")
        raw = provider.generate_json(
            system=_SOURCE_SYSTEM,
            user=json.dumps({
                "asOfDateExclusive": packet["asOfDateExclusive"],
                "source": {
                    "sourceId": source_id,
                    "sourceDate": source["sourceDate"],
                    "claims": source["claims"],
                },
                "validatedModelChanges": grounded,
                "historicalPITVerified": False,
            }, ensure_ascii=False),
            schema=SourceExplanation.model_json_schema(),
        )
        diagnostics.append(getattr(provider, "last_generation_diagnostics", None))
        parsed = SourceExplanation.model_validate_json(raw)
        if (
            parsed.sourceId != source_id
            or (parsed.sourceId, parsed.claimId) not in available
            or not _complete(parsed.explanation)
            or not _complete(parsed.limitation)
            or is_observed_market_result(parsed.explanation)
            or is_observed_market_result(parsed.limitation)
        ):
            raise ComposedEvidenceError("COMPOSED_SOURCE_EXPLANATION_INVALID")
        explanations.append(parsed.model_dump())
    narrative = " ".join(x["explanation"] for x in explanations)
    limitations = " ".join(x["limitation"] for x in explanations)
    if len(narrative) > 1200 or len(limitations) > 1200:
        raise ComposedEvidenceError("COMPOSED_NARRATIVE_LENGTH_EXCEEDED")
    return {
        "sourceExplanations": explanations,
        "sourceGenerationDiagnostics": diagnostics,
        "whatChanged": narrative,
        "whatRemainsUnknown": limitations,
        "sourceCoverage": {
            "requiredSourceIds": [x["sourceId"] for x in new_sources],
            "explainedSourceIds": [x["sourceId"] for x in explanations],
        },
        "narrativeAssembly": "VERBATIM_SOURCE_MODEL_SENTENCES",
        "modelTextRewritten": False,
        "semanticAccuracyVerified": False,
        "historicalPITVerified": False,
    }


def assess_question_axes(
    provider: LLMProvider, packet: dict, draft: dict,
) -> dict:
    """Do not derive either dimension from legacy OPEN/RESOLVED labels."""
    questions = draft["visibleQuestions"]
    if not questions:
        return {"questionAssessments": [], "generationDiagnostics": None}
    raw = provider.generate_json(
        system=_QUESTIONS_SYSTEM,
        user=json.dumps({
            "asOfDateExclusive": packet["asOfDateExclusive"],
            "visibleOfficialSources": [
                {"sourceId": x["sourceId"], "sourceDate": x["sourceDate"],
                 "claims": x["claims"]}
                for x in packet["eligibleSources"]
            ],
            "visibleQuestions": questions,
            "pointInTimeVerified": False,
        }, ensure_ascii=False),
        schema=AllQuestionAxes.model_json_schema(),
    )
    diag = getattr(provider, "last_generation_diagnostics", None)
    parsed = AllQuestionAxes.model_validate_json(raw)
    states = [x.model_dump() for x in parsed.assessments]
    expected = {x["questionId"] for x in questions}
    if (
        len(states) != len(questions)
        or {x["questionId"] for x in states} != expected
        or any(
            (x["sourceId"], x["claimId"]) not in _refs(packet)
            or not _complete(x["explanation"])
            or not _complete(x["remainingEvidence"])
            or is_observed_market_result(x["explanation"])
            for x in states
        )
    ):
        raise ComposedEvidenceError("COMPOSED_QUESTION_AXES_INVALID")
    return {
        "questionAssessments": states,
        "generationDiagnostics": diag,
        "axesSemanticallyVerified": False,
        "legacyStatusAutomaticallyOverwritten": False,
        "humanReviewRequired": True,
    }
