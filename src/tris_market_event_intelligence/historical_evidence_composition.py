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
    # Provenance is attached from the one-claim input envelope, never
    # model-invented identifiers. Both text fields remain model-authored.
    explanation: str = Field(min_length=35, max_length=310)
    limitation: str = Field(min_length=25, max_length=230)


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


class QuestionAnswer(Strict):
    regulatoryActionProgress: Literal[
        "NOT_OBSERVED", "ANNOUNCED", "AUTHORIZED", "IMPLEMENTING",
        "COMPLETED", "UNKNOWN",
    ]
    requestedEvidenceStatus: Literal[
        "NOT_OBSERVED", "PARTIAL", "DIRECTLY_OBSERVED",
    ]
    explanation: str = Field(min_length=35, max_length=450)
    remainingEvidence: str = Field(min_length=25, max_length=280)




_SOURCE_SYSTEM = """Explain ONLY the single dated official claim given.
Write one concise complete factual sentence describing the regulator,
decision or action and its exact aircraft/product scope. Distinguish
recommendation, authorization, implementation and completion. Then write
one concise limitation describing exactly what this claim does not
establish. No future facts, economic effects, stock prices, inferred
causality, unobserved results or IDs in prose. Both sentences must end
with punctuation. Return ONLY the two schema fields: explanation and
limitation. The invoking system attaches the source and claim IDs from
the cutoff-visible official record; you must not generate those IDs."""

_SCREEN_SYSTEM = """For ONE historical question, classify EACH supplied
date-cutoff-visible official claim, with one role per claim index.
DIRECT_ANSWER means this claim directly answers the exact question.
INTERMEDIATE_DECISION means an official later decision or significant
regulatory step bears on the question, but does not supply all requested
records/outcomes. RELATED_CONTEXT means preliminary background or
evidence explaining the question. NOT_RELEVANT means unrelated scope.
A decision restricting production expansion is an INTERMEDIATE_DECISION
when the question asks about official production oversight decisions AND
completed audit findings; lack of final audit results does not erase the
restriction. Likewise, product authorization can be an INTERMEDIATE_DECISION
for a question about underlying manufacturer filings and internal review
records; approval does not itself supply those records. These are general
distinctions, not proof of any particular historical outcome.
Classify later decisions separately from earlier notices. Preserve distinct
product and aircraft scopes. Do not use later unseen information. Return
exactly the required claim-index keys and allowed role values. No prose."""


_ANSWER_SYSTEM = """Use the question and all model-screened relevant
official claims to answer the exact historical question as of the cutoff.
INTERMEDIATE_DECISION claims are genuine regulatory actions; describe them
even when the final audit findings, manufacturer filings or underlying
review records are not visible. They do NOT establish completed inspections,
completed audits, completed implementation or the presence of private
internal documentation. DIRECT_ANSWER supports requestedEvidenceStatus
DIRECTLY_OBSERVED only when the entire requested answer is supported.
When observed intermediate decisions exist but the requested final records
remain unseen, use PARTIAL rather than NOT_OBSERVED.
regulatoryActionProgress is about the regulatory action in the question's
scope. AUTHORIZED means an approval/authorization was issued. COMPLETED
requires an explicit source establishing completion of the relevant
implementation or final process, not merely authorization.
Distinguish regulator approvals from product adoption, execution,
inspection completion or factory audit completion. No future sources,
unsupported causes, prices or financial market claims. Provide complete
explanation and remainingEvidence sentences and strict JSON only."""


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
    changes = draft["modelAuthoredChangeItems"]
    explanations: list[dict] = []
    diagnostics: list[dict] = []
    for source in new_sources:
        source_id = source["sourceId"]
        grounded = [x for x in changes if x["sourceId"] == source_id]
        if not grounded:
            raise ComposedEvidenceError("COMPOSED_SOURCE_CHANGE_MISSING")
        source_diagnostics: dict = {
            "sourceId": source_id, "claimCalls": [],
        }
        for claim in source["claims"]:
            claim_id = claim["claimId"]
            relevant_changes = [
                x for x in grounded if x["claimId"] == claim_id
            ]
            # The full original official claim is the evidence boundary.
            # The stage-one change is supplementary model text, never proof.
            raw = provider.generate_json(
                system=_SOURCE_SYSTEM,
                user=json.dumps({
                    "asOfDateExclusive": packet["asOfDateExclusive"],
                    "sourceId": source_id,
                    "sourceDate": source["sourceDate"],
                    "claim": claim,
                    "modelChangeForClaim": relevant_changes,
                    "historicalPITVerified": False,
                }, ensure_ascii=False),
                schema=SourceExplanation.model_json_schema(),
            )
            generation = getattr(provider, "last_generation_diagnostics", None)
            parsed = SourceExplanation.model_validate_json(raw)
            if not _complete(parsed.explanation):
                raise ComposedEvidenceError(
                    "COMPOSED_SOURCE_EXPLANATION_INCOMPLETE:"
                    + source_id + ":" + claim_id
                )
            if not _complete(parsed.limitation):
                raise ComposedEvidenceError(
                    "COMPOSED_SOURCE_LIMITATION_INCOMPLETE:"
                    + source_id + ":" + claim_id
                )
            if (is_observed_market_result(parsed.explanation)
                    or is_observed_market_result(parsed.limitation)):
                raise ComposedEvidenceError(
                    "COMPOSED_SOURCE_UNSUPPORTED_OUTCOME:"
                    + source_id + ":" + claim_id
                )
            # IDs are authoritative input metadata, not generated model claims.
            explanations.append({
                "sourceId": source_id,
                "claimId": claim_id,
                **parsed.model_dump(),
            })
            source_diagnostics["claimCalls"].append({
                "claimId": claim_id, "provider": generation,
            })
        diagnostics.append(source_diagnostics)
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
        "narrativeAssembly": "VERBATIM_CLAIM_MODEL_SENTENCES",
        "claimProvenance": "ASSIGNED_FROM_INPUT_NOT_MODEL_GENERATED",
        "modelTextRewritten": False,
        "semanticAccuracyVerified": False,
        "historicalPITVerified": False,
    }


def assess_question_axes(
    provider: LLMProvider, packet: dict, draft: dict,
    composed: dict | None = None,
) -> dict:
    """Screen every visible official claim per question, then synthesize.

    A claim's role is model-authored; provenance and order are input-bound.
    Full screening decisions are persisted for human recall/omission review.
    No curator transitions or independently unverified outcome labels are
    supplied, and structural completeness does not prove semantic relevance.
    """
    questions = draft["visibleQuestions"]
    if not questions:
        return {
            "questionAssessments": [], "generationDiagnostics": None,
            "questionEvidenceScreens": [],
        }
    visible: list[dict] = [
        {
            "sourceId": source["sourceId"],
            "claimId": claim["claimId"],
            "sourceDate": source["sourceDate"],
            "text": claim["text"],
        }
        for source in packet["eligibleSources"]
        for claim in source["claims"]
    ]
    explanations = {
        (item["sourceId"], item["claimId"]): item["explanation"]
        for item in (composed["sourceExplanations"] if composed else [])
    }
    screen_roles = (
        "DIRECT_ANSWER", "INTERMEDIATE_DECISION", "RELATED_CONTEXT",
        "NOT_RELEVANT",
    )
    screened: list[dict] = []
    assessments: list[dict] = []
    screen_diagnostics: list[dict | None] = []
    answer_diagnostics: list[dict | None] = []
    keys = [f"claim_{i}" for i in range(len(visible))]
    screen_schema: dict = {
        "type": "object",
        "properties": {
            key: {"type": "string", "enum": list(screen_roles)}
            for key in keys
        },
        "required": keys,
        "additionalProperties": False,
    }
    for question in questions:
        raw_screen = provider.generate_json(
            system=_SCREEN_SYSTEM,
            user=json.dumps({
                "asOfDateExclusive": packet["asOfDateExclusive"],
                "question": question,
                "indexedVisibleOfficialClaims": [
                    {"index": i, **claim} for i, claim in enumerate(visible)
                ],
                "pointInTimeVerified": False,
            }, ensure_ascii=False),
            schema=screen_schema,
        )
        screen_diagnostics.append(getattr(
            provider, "last_generation_diagnostics", None,
        ))
        try:
            roles = json.loads(raw_screen)
        except (ValueError, TypeError) as exc:
            raise ComposedEvidenceError(
                "COMPOSED_QUESTION_SCREEN_INVALID_JSON"
            ) from exc
        if (
            not isinstance(roles, dict)
            or set(roles) != set(keys)
            or any(roles[key] not in screen_roles for key in keys)
        ):
            raise ComposedEvidenceError(
                "COMPOSED_QUESTION_SCREEN_INCOMPLETE"
            )
        decisions = [
            {
                "sourceId": claim["sourceId"],
                "claimId": claim["claimId"],
                "sourceDate": claim["sourceDate"],
                "role": roles[f"claim_{i}"],
            }
            for i, claim in enumerate(visible)
        ]
        relevant = [
            claim for claim in decisions if claim["role"] != "NOT_RELEVANT"
        ]
        if not relevant:
            raise ComposedEvidenceError(
                "COMPOSED_QUESTION_NO_RELEVANT_EVIDENCE"
            )
        # Most recent strongest applicable action is chosen deterministically
        # from model-screened roles. This is a provenance tie-breaker, not
        # independent evidence that the relevance classification is correct.
        priority = {
            "DIRECT_ANSWER": 3, "INTERMEDIATE_DECISION": 2,
            "RELATED_CONTEXT": 1,
        }
        primary = max(
            relevant,
            key=lambda item: (
                priority[item["role"]], item["sourceDate"],
                next(i for i, claim in enumerate(visible)
                     if claim["sourceId"] == item["sourceId"]
                     and claim["claimId"] == item["claimId"]),
            ),
        )
        candidate_input = [
            {
                **claim,
                "role": decision["role"],
                "sourceScopedExplanation": explanations.get(
                    (claim["sourceId"], claim["claimId"]),
                ),
            }
            for claim, decision in zip(visible, decisions)
            if decision["role"] != "NOT_RELEVANT"
        ]
        raw_answer = provider.generate_json(
            system=_ANSWER_SYSTEM,
            user=json.dumps({
                "asOfDateExclusive": packet["asOfDateExclusive"],
                "question": question,
                "screenedRelevantEvidence": candidate_input,
                "historicalPITVerified": False,
            }, ensure_ascii=False),
            schema=QuestionAnswer.model_json_schema(),
        )
        answer_diagnostics.append(getattr(
            provider, "last_generation_diagnostics", None,
        ))
        answer = QuestionAnswer.model_validate_json(raw_answer)
        invalid = (
            "EXPLANATION_INCOMPLETE" if not _complete(answer.explanation)
            else "REMAINING_EVIDENCE_INCOMPLETE"
            if not _complete(answer.remainingEvidence)
            else "EXPLANATION_MARKET_OUTCOME"
            if is_observed_market_result(answer.explanation)
            else "REMAINING_EVIDENCE_MARKET_OUTCOME"
            if is_observed_market_result(answer.remainingEvidence)
            else None
        )
        if invalid is not None:
            # Never alter model prose or convert an invalid answer to success.
            # Include the question and failed field so genuine Ollama failures
            # are actionable without logging entire raw model responses.
            raise ComposedEvidenceError(
                "COMPOSED_QUESTION_ANSWER_INVALID:"
                + question["questionId"] + ":" + invalid
            )
        if (
            answer.requestedEvidenceStatus == "DIRECTLY_OBSERVED"
            and not any(x["role"] == "DIRECT_ANSWER" for x in relevant)
        ):
            raise ComposedEvidenceError(
                "COMPOSED_QUESTION_DIRECT_ANSWER_UNSUPPORTED"
            )
        resolved = {
            "questionId": question["questionId"],
            "sourceId": primary["sourceId"],
            "claimId": primary["claimId"],
            "supportingEvidence": [
                {"sourceId": x["sourceId"], "claimId": x["claimId"]}
                for x in relevant
            ],
            **answer.model_dump(),
        }
        QuestionAxes.model_validate(resolved)
        assessments.append(resolved)
        screened.append({
            "questionId": question["questionId"],
            "allVisibleClaimRoles": decisions,
            "primaryClaimSelection": {
                "sourceId": primary["sourceId"],
                "claimId": primary["claimId"],
                "role": primary["role"],
            },
            "roleAccuracyVerified": False,
            "omissionsSemanticallyVerified": False,
        })
    return {
        "questionAssessments": assessments,
        "questionEvidenceScreens": screened,
        "generationDiagnostics": {
            "relevanceScreenCalls": screen_diagnostics,
            "questionAnswerCalls": answer_diagnostics,
        },
        "axesSemanticallyVerified": False,
        "legacyStatusAutomaticallyOverwritten": False,
        "humanReviewRequired": True,
    }
