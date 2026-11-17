"""Multi-event reasoning: atomic evidence, event chronology, mechanism abstention."""
from __future__ import annotations

import json

import pytest

from test_historical_news_impact_test import sample_case

from tris_market_event_intelligence import historical_news_impact_test_cli as news
from tris_market_event_intelligence.historical_reasoning_study_cli import (
    compare_cases,
    study_case,
)


def _case(kind: str) -> news.CaseInput:
    data = sample_case()
    data["caseId"] = f"research-{kind}"
    data["knownTickers"] = []
    if kind == "energy":
        data["articles"][0].update({
            "title": "EnergyWorks reported pipeline maintenance and a power outage",
            "body": (
                "EnergyWorks reported pipeline maintenance and a regional "
                "power outage. Operators stated that a mechanical fault "
                "interrupted gas deliveries. Customers were notified of "
                "potential delays in service."
            ),
        })
        data["articles"][1].update({
            "title": "EnergyWorks updated outage investigation with new details",
            "body": (
                "EnergyWorks said the pipeline remained offline for repairs. "
                "Officials stated that a fault investigation was underway. "
                "No regulatory sanctions had been announced."
            ),
        })
    elif kind == "transport":
        data["articles"][0].update({
            "title": "HarborTransit reported disruptions during a port labor dispute",
            "body": (
                "HarborTransit reported a labor dispute delaying port cargo "
                "handling. The company said several scheduled shipments "
                "could face delays. Negotiations continued without agreement."
            ),
        })
        data["articles"][1].update({
            "title": "HarborTransit updated shipping operations after negotiations",
            "body": (
                "HarborTransit confirmed that port cargo operations remained "
                "restricted. A spokesperson reported continued talks with "
                "workers. Other ports were not confirmed to be affected."
            ),
        })
    return news.CaseInput.model_validate(data)


def _response(case: news.CaseInput, *, contaminated: bool = False) -> dict:
    _, lookup = news._quote_anchors(case)
    a1, a2 = lookup["a1:q1"]["text"], lookup["a2:q1"]["text"]
    # Two independently referenced atomic spans from one source quotation.
    first = a1.split(" ", 6)
    excerpt1 = " ".join(first[:6])
    excerpt2 = " ".join(first[3:])
    # Ensure a second distinct claim from the same original source excerpt.
    return {
        "atomicClaims": [
            {
                "claimId": "fact1", "articleId": "a1", "quoteId": "a1:q1",
                "excerpt": excerpt1, "kind": "FACT", "eventStage": "INITIAL_REPORT",
            },
            {
                "claimId": "fact2", "articleId": "a1", "quoteId": "a1:q1",
                "excerpt": excerpt2, "kind": "FACT", "eventStage": "NEW_FACT",
            },
            {
                "claimId": "fact3", "articleId": "a2", "quoteId": "a2:q1",
                "excerpt": a2, "kind": "FACT", "eventStage": "REACTION",
            },
        ],
        "eventEdges": [
            {
                "earlierClaimId": "fact1", "laterClaimId": "fact3",
                "relation": "FOLLOW_UP",
                "rationale": "The later report provides more context on the earlier event.",
            },
        ],
        "economicMechanisms": [
            {
                "targetType": "COMPANY",
                "target": (
                    "EnergyWorks" if "energy" in case.caseId
                    else "HarborTransit" if "transport" in case.caseId
                    else "Example Devices"
                ),
                "direction": "UNCERTAIN",
                "evidenceClaimIds": (
                    ["fact1", "fact2"] if not contaminated else ["fact9"]
                ),
                "causalClaim": {
                    "triggerCondition": (
                        "If the reported disruption persists into the next quarter"
                    ),
                    "transmissionChannel": "REVENUE",
                    "potentialEffect": (
                        "Company sales could suffer if customers delay purchases."
                    ),
                    "counterHypothesis": (
                        "If the issue resolves quickly, revenue may remain stable."
                    ),
                },
            }
        ],
    }


class MockResearcher:
    model_id = "offline-atomic-cross-event"

    def __init__(self, case: news.CaseInput, *, bad: bool = False):
        self.case = case
        self.bad = bad
        self.calls = 0

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        self.calls += 1
        payload = json.loads(user)
        assert "atomicClaims" in schema["properties"]
        assert payload["historicalOutcomesAvailable"] is False
        assert payload["observedPricesAvailable"] is False
        assert "actualStockReturnPct" not in user
        assert "3.1%" not in user
        return json.dumps(_response(self.case, contaminated=self.bad))


def test_atomic_decomposition_multiple_claims_same_quote_still_verifies() -> None:
    case = _case("production")
    provider = MockResearcher(case)
    result = study_case(case, provider)
    assert provider.calls == 1
    assert len(result["atomicClaims"]) == 3
    assert result["atomicClaims"][0]["quoteId"] == result["atomicClaims"][1]["quoteId"]
    assert len(result["eventEdges"]) == 1
    assert len(result["economicMechanisms"]) == 1
    assert result["economicMechanisms"][0]["claimVerification"]["status"] == (
        "SOURCE_QUOTE_MATCHED_CONDITIONAL"
    )
    assert result["causalityProven"] is False
    assert result["historicalBenchmarkEligible"] is False


def test_three_distinct_event_domains_only_prove_structural_coverage() -> None:
    cases = [_case(k) for k in ("production", "energy", "transport")]
    outcomes = [study_case(c, MockResearcher(c)) for c in cases]
    review = compare_cases(outcomes)
    assert review["caseCount"] == 3
    assert set(review["caseIds"]) == {
        "research-production", "research-energy", "research-transport",
    }
    assert all(x == 3 for x in review["atomicClaimCounts"].values())
    assert review["generalizationVerified"] is False
    assert review["outcomeAccuracyVerified"] is False
    assert not any(row["marketPricesRead"] for row in outcomes)


def test_missing_atomic_evidence_fails_closed_without_invalidating_event() -> None:
    case = _case("production")
    report = study_case(case, MockResearcher(case, bad=True))
    assert report["eventEdges"]
    assert not report["economicMechanisms"]
    assert report["rejectedEconomicMechanisms"][0]["reason"] == (
        "UNSUPPORTED_ATOMIC_CLAIM_IDS"
    )


def test_observed_market_text_withheld_from_model() -> None:
    case_data = _case("energy").model_dump()
    case_data["articles"][1]["body"] += (
        " Shares dropped 3.1% on the day and the market declined."
    )
    case = news.CaseInput.model_validate(case_data)
    report = study_case(case, MockResearcher(case))
    assert report["withheldMarketOutcomeQuotes"]
    assert report["marketPricesRead"] is False


def test_cross_event_comparison_refuses_duplicate_case_ids() -> None:
    case = _case("production")
    r = study_case(case, MockResearcher(case))
    with pytest.raises(ValueError, match="DISTINCT_CASES"):
        compare_cases([r, r])
