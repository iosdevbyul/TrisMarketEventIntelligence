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


class QuestionEvidenceRef(Strict):
    sourceId: str
    claimId: str


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
    supportingEvidence: list[QuestionEvidenceRef] = Field(min_length=1, max_length=12)
    explanation: str = Field(min_length=35, max_length=450)
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

_QUESTIONS_SYSTEM = """Evaluate every visible official source claim and
every supplied source-scoped explanation relevant to EACH research question.
Later official decisions may change earlier UNKNOWN or OPEN answers. Prior
model assessments are provisional, not ground truth. Include EVERY directly
relevant dated claim in supportingEvidence, including new announcements and
older pending notices, without adding unrelated or invented references.
The primary sourceId and claimId should cite the most recent substantive
relevant decision, rather than always citing the initial question source.

Assess two independent dimensions. regulatoryActionProgress measures the
regulator action (NOT_OBSERVED, ANNOUNCED, AUTHORIZED, IMPLEMENTING,
COMPLETED, UNKNOWN), not completion of inspections or audits. An official
production restriction is an observed regulatory decision but not a
completed quality audit. requestedEvidenceStatus measures whether the
EXACT full question has supporting documentation: NOT_OBSERVED when no
part of the answer is visible, PARTIAL when a decision or intermediate
evidence exists but necessary final results are missing, and
DIRECTLY_OBSERVED when the full answer requested is directly observed.
For questions requesting BOTH official decisions and audit findings,
observed official restrictions plus missing audit findings mean PARTIAL,
not NOT_OBSERVED. Explain the observed decision in explanation, then
name only truly absent evidence in remainingEvidence. Never claim no
decisions are visible when a relevant decision is in supportingEvidence.
Distinguish 737-900ER from 737-9 MAX and authorization from actual
completion. No editorial future transitions, fabricated proof, market
prices, financial effects or causal assertions. Strict JSON only."""


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
    composed: dict | None = None,
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
            "newlyExplainedClaims": (
                composed["sourceExplanations"] if composed is not None else []
            ),
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
    visible_refs = _refs(packet)
    if (
        len(states) != len(questions)
        or {x["questionId"] for x in states} != expected
        or any(
            (x["sourceId"], x["claimId"]) not in visible_refs
            or (x["sourceId"], x["claimId"]) not in {
                (ref["sourceId"], ref["claimId"])
                for ref in x["supportingEvidence"]
            }
            or len({
                (ref["sourceId"], ref["claimId"])
                for ref in x["supportingEvidence"]
            }) != len(x["supportingEvidence"])
            or any(
                (ref["sourceId"], ref["claimId"]) not in visible_refs
                for ref in x["supportingEvidence"]
            )
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
