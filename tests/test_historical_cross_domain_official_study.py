"""Real distinct official 2023-24 events, evidence and abstention safeguards."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

from tris_market_event_intelligence import historical_cross_domain_official_study_cli as cross

COHORT = Path("research/cross-domain-official-events-2023-2024.json")


def cohort() -> dict:
    return json.loads(COHORT.read_text(encoding="utf-8"))


class FakeProvider:
    model_id = "mock-real-three-domain"

    def __init__(self, *, invent_target: bool = False,
                 invent_source: bool = False, invented_outcome: bool = False):
        self.calls = 0
        self.invent_target = invent_target
        self.invent_source = invent_source
        self.invented_outcome = invented_outcome

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        self.calls += 1
        packet = json.loads(user)
        assert packet["realizedPricesOrOutcomesAvailable"] is False
        assert packet["historicalPageVersionVerified"] is False
        assert packet["publishedAtTimeVerified"] is False
        assert "hypotheses" in schema["properties"]
        facts = packet["sourceFactParaphrases"]
        proposals = []
        if packet["allowedEconomicTargets"] or self.invent_target:
            is_faa = packet["category"] == "AVIATION_SAFETY_REGULATION"
            target = (
                packet["allowedEconomicTargets"][0]
                if packet["allowedEconomicTargets"] and not self.invent_target
                else "ImaginaryPublicCo"
            )
            statement = (
                "If airlines must complete the mandated inspections"
                if is_faa else "If authorized vaccines become available"
            )
            proposal = {
                "target": target,
                "sourceFactIds": [
                    "invented" if self.invent_source else facts[1]["factId"]
                ],
                "transmission": "REGULATION" if is_faa else "DEMAND",
                "direction": "UNCERTAIN",
                "mechanismJustification": (
                    "The official government action changes the operating "
                    "conditions under which the named participant may operate."
                ),
                "triggerCondition": statement,
                "conditionalEffect": (
                    "The company could face additional compliance costs "
                    "if mandatory safety inspections delay operations."
                    if is_faa else
                    "Vaccine demand could change if recipients choose the updated doses."
                ),
                "counterHypothesis": (
                    "If the change has little effect on behavior, "
                    "economic consequences might remain small."
                ),
                "missingEvidence": (
                    "Independent information about company orders "
                    "and actual market participation is not provided."
                ),
                "evidenceBoundary": (
                    "The agency action is documented but future economic "
                    "results and realized company sales are not supplied."
                ),
            }
            if self.invented_outcome:
                proposal["conditionalEffect"] = (
                    "Company shares fell 4% following the announcement."
                )
            proposals.append(proposal)
        return json.dumps({
            "economicDecision": "PROPOSE" if proposals else "ABSTAIN",
            "factAssessments": [
                {
                    "factId": fact["factId"],
                    "interpretation": (
                        "This curator statement describes an agency event "
                        "and is not evidence of a stock market outcome."
                    ),
                    "limitation": (
                        "The official date does not establish an independently "
                        "archived first publication time."
                    ),
                }
                for fact in facts
            ],
            "hypotheses": proposals,
            "abstentionRationale": (
                "No identifiable market-exposed target is documented, "
                "so this event does not ground a commercial hypothesis."
                if not proposals else
                "Possible commercial transmission is hypothetical "
                "and does not establish a realized market result."
            ),
            "overallInterpretation": (
                "Official historical facts are reported, but the "
                "economic significance remains unverified and time-limited."
            ),
        })


def test_official_three_event_cohort_spans_independent_domains() -> None:
    doc = cohort()
    cross.validate_cohort(doc)
    assert [c["eventDate"] for c in doc["cases"]] == [
        "2024-01-06", "2023-09-11", "2023-09-24",
    ]
    assert {c["source"]["publisher"] for c in doc["cases"]} == {
        "Federal Aviation Administration",
        "US Food and Drug Administration",
        "National Aeronautics and Space Administration",
    }
    assert all(c["source"]["dateOnly"] for c in doc["cases"])
    assert doc["cases"][0]["source"]["retrospectivePage"] is True
    assert doc["cases"][2]["allowedEconomicTargets"] == []


def test_actual_three_domain_research_respects_science_abstention() -> None:
    fake = FakeProvider()
    result = cross.study_cohort(cohort(), fake)
    assert fake.calls == 3
    assert result["modelCalls"] == 3
    assert result["review"]["sourceFactAssessments"] == 9
    assert result["review"]["differentEventDomains"] == 3
    assert result["review"]["distinctOfficialSources"] == 3
    assert result["review"]["negativeControlAbstained"] is True
    assert result["review"]["researchCandidateCount"] == 2
    assert result["review"]["rejectedHypothesisCount"] == 0
    assert result["historicalPublisherTimeVerified"] is False
    assert result["actualPricesRead"] is False
    assert result["promotionAllowed"] is False
    assert result["trainingPerformed"] is False


def test_unsupported_company_or_fact_not_promoted() -> None:
    doc = cohort()
    bad_company = cross.study_case(doc["cases"][2], FakeProvider(invent_target=True))
    assert bad_company["negativeControlAbstention"] is False
    assert bad_company["researchCandidates"] == []
    assert bad_company["rejectedHypotheses"][0]["reasonCodes"] == [
        "TARGET_NOT_SUPPORTED_BY_OFFICIAL_SOURCE"
    ]
    wrong_fact = cross.study_case(doc["cases"][0], FakeProvider(invent_source=True))
    assert wrong_fact["researchCandidates"] == []
    assert wrong_fact["rejectedHypotheses"][0]["reasonCodes"] == [
        "UNKNOWN_OR_DUPLICATE_SOURCE_FACT"
    ]


def test_hallucinated_observed_market_result_blocked() -> None:
    result = cross.study_case(
        cohort()["cases"][0], FakeProvider(invented_outcome=True),
    )
    assert not result["researchCandidates"]
    assert "OBSERVED_MARKET_RESULT_NOT_SUPPORTED" in (
        result["rejectedHypotheses"][0]["reasonCodes"]
    )


def test_no_extra_domains_or_unsafe_targets_in_research_input() -> None:
    doc = cohort()
    changed = copy.deepcopy(doc)
    changed["cases"][1]["source"]["url"] = "http://localhost/private"
    with pytest.raises(ValueError, match="SOURCE_INVALID"):
        cross.validate_cohort(changed)
    changed = copy.deepcopy(doc)
    changed["cases"][2]["allowedEconomicTargets"] = ["ImaginaryPublicCo"]
    with pytest.raises(ValueError, match="NEGATIVE_CONTROL"):
        cross.validate_cohort(changed)
    changed = copy.deepcopy(doc)
    changed["cases"][2]["category"] = changed["cases"][0]["category"]
    with pytest.raises(ValueError, match="CATEGORY"):
        cross.validate_cohort(changed)


def test_source_fact_id_coverage_is_mandatory() -> None:
    class DuplicateFact(FakeProvider):
        def generate_json(self, *, system: str, user: str, schema: dict) -> str:
            response = json.loads(super().generate_json(
                system=system, user=user, schema=schema,
            ))
            response["factAssessments"][1]["factId"] = (
                response["factAssessments"][0]["factId"]
            )
            return json.dumps(response)

    with pytest.raises(ValueError, match="FACT_COVERAGE_FAILED"):
        cross.study_case(cohort()["cases"][0], DuplicateFact())


def test_preview_reads_real_case_cohort_without_ollama_or_creating_file(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", [
        "historical-cross-domain", "--cohort", str(COHORT), "--preview",
    ])
    cross.main()
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "THREE_CROSS_DOMAIN_OFFICIAL_EVENTS_READY"
    assert result["officialSourceCount"] == 3
    assert result["curatedFacts"] == 9
    assert result["negativeControls"] == 1
    assert result["historicalPointInTimeVerified"] is False
    assert result["modelCalls"] == 0
