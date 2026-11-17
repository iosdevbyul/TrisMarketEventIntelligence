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


class SourceExplanations(Strict):
    items: list[SourceExplanation] = Field(min_length=1, max_length=12)


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


_SOURCE_SYSTEM = """Explain EACH listed claim of ONE dated official source,
producing exactly one short item per claim with its exact sourceId and
claimId. Every claim is required: DO NOT select only the most salient one.
Explain all separately announced actions, even if they share one release.
Use only source claims and already validated model change assertions.
Name regulator, action type and relevant product/aircraft scope. Distinguish
recommendation, authorization, implementation and completion. Do not
invent results, costs, price outcomes or future data. No opaque IDs in
prose. Identify what each claim does not establish in its limitation.
All prose must end in full-sentence punctuation. Return strict JSON."""

_QUESTIONS_SYSTEM = """Assess each research question ONLY against the
provided date-cutoff-visible official sources, not future transitions.
Give exactly one row per question. The axes are INDEPENDENT. Consider ALL eligible dated evidence, especially
later claims that update earlier pending decisions. The provided prior
MODEL-authored question assessments are provisional, not ground truth;
compare their cited evidence with all eligible sources. Anchor a resolved
question in the NEWEST directly relevant official claim, not the first
claim that introduced it. A question asking whether a PROCEDURE was
authorized can be directly answered by its official authorization
announcement even if individual work remains incomplete.
The axes are INDEPENDENT:
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
    """Generate and validate one explanation per new source claim, then compose.

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
        schema = SourceExplanations.model_json_schema()
        schema["properties"]["items"]["minItems"] = len(source["claims"])
        schema["properties"]["items"]["maxItems"] = len(source["claims"])
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
            schema=schema,
        )
        diagnostics.append(getattr(provider, "last_generation_diagnostics", None))
        parsed = SourceExplanations.model_validate_json(raw)
        items = [x.model_dump() for x in parsed.items]
        required = {(source_id, x["claimId"]) for x in source["claims"]}
        submitted = [(x["sourceId"], x["claimId"]) for x in items]
        if (
            len(submitted) != len(required)
            or set(submitted) != required
            or len(set(submitted)) != len(submitted)
            or any(
                (x["sourceId"], x["claimId"]) not in available
                or not _complete(x["explanation"])
                or not _complete(x["limitation"])
                or is_observed_market_result(x["explanation"])
                or is_observed_market_result(x["limitation"])
                for x in items
            )
        ):
            raise ComposedEvidenceError("COMPOSED_SOURCE_CLAIM_COVERAGE_INVALID")
        indexed = {x["claimId"]: x for x in items}
        explanations.extend(indexed[claim["claimId"]] for claim in source["claims"])
    narrative = " ".join(x["explanation"] for x in explanations)
    limitations = " ".join(x["limitation"] for x in explanations)
    if len(narrative) > 2400 or len(limitations) > 2400:
        raise ComposedEvidenceError("COMPOSED_NARRATIVE_LENGTH_EXCEEDED")
    return {
        "sourceExplanations": explanations,
        "sourceGenerationDiagnostics": diagnostics,
        "whatChanged": narrative,
        "whatRemainsUnknown": limitations,
        "sourceCoverage": {
            "requiredSourceIds": [x["sourceId"] for x in new_sources],
            "explainedSourceIds": list(dict.fromkeys(
                x["sourceId"] for x in explanations
            )),
            "requiredClaimRefs": [
                {"sourceId": source["sourceId"], "claimId": claim["claimId"]}
                for source in new_sources for claim in source["claims"]
            ],
            "explainedClaimRefs": [
                {"sourceId": x["sourceId"], "claimId": x["claimId"]}
                for x in explanations
            ],
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
    schema = AllQuestionAxes.model_json_schema()
    schema["properties"]["assessments"]["minItems"] = len(questions)
    schema["properties"]["assessments"]["maxItems"] = len(questions)
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
            "previouslyValidatedModelQuestionAssessments": (
                draft["modelAuthoredQuestionAssessments"]
            ),
            "pointInTimeVerified": False,
        }, ensure_ascii=False),
        schema=schema,
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
