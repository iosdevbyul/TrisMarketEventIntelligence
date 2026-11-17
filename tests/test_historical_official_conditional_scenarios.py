"""Real Gemma4 v3 failure-shape regression: conditional scenarios vs form."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from test_historical_official_fact_reasoning import SourceFactProvider

from tris_market_event_intelligence import historical_official_fact_reasoning_cli as official
from tris_market_event_intelligence.historical_official_conditional_scenarios import (
    review_scenario,
)

COHORT = Path("research/independent-historical-news-2023.json")

# Faithful examples from the observed v3 failure output, with unchanged
# economically significant phrases. These are MODEL CLAIMS, not ground truth.
V3 = [
    {
        "sourceFactIds": ["msft-azure", "msft-cloud"],
        "transmission": "REVENUE", "direction": "POSITIVE",
        "triggerCondition": (
            "Increased customer adoption and utilization of Azure "
            "and Microsoft Cloud services."
        ),
        "conditionalEffect": (
            "If customer demand for cloud infrastructure and software "
            "services remains robust, Microsoft could continue to realize "
            "revenue growth rates similar to those reported (26% and 21% YoY)."
        ),
        "counterHypothesis": (
            "If macroeconomic conditions lead to reduced enterprise IT "
            "spending or increased competitive pressure from alternative "
            "cloud providers, the growth rates for Azure and Microsoft "
            "Cloud revenue could decelerate."
        ),
        "missingEvidence": (
            "Data on enterprise IT spending trends, competitive market "
            "share shifts and future cloud service uptake guidance."
        ),
    },
    {
        "sourceFactIds": ["googl-cloud", "googl-operating"],
        "transmission": "REVENUE", "direction": "POSITIVE",
        "triggerCondition": "Continued growth in Google Cloud revenue.",
        "conditionalEffect": (
            "If Google Cloud revenue continues to grow at a rate comparable "
            "to the year-over-year increase observed, it could lead to "
            "increased operating income, provided that the associated costs "
            "(COSTS) do not increase disproportionately."
        ),
        "counterHypothesis": (
            "If the growth in revenue is driven by significant increases "
            "in infrastructure costs or if customer uptake slows down, "
            "the operating income could decline or revert to a loss."
        ),
        "missingEvidence": (
            "Data on the cost structure of Google Cloud for Q2 2023 "
            "and subsequent periods to assess operating profit sustainability."
        ),
    },
    {
        "sourceFactIds": ["amd-revenue", "amd-gaap"],
        "transmission": "REVENUE", "direction": "NEGATIVE",
        "triggerCondition": (
            "If the 18 percent year-over-year revenue decline is due to "
            "reduced customer uptake or market saturation."
        ),
        "conditionalEffect": (
            "AMD's operating margin could be further pressured if the "
            "decline in revenue is not offset by cost efficiencies."
        ),
        "counterHypothesis": (
            "The revenue decline is primarily due to a cyclical downturn "
            "in the broader semiconductor market, rather than AMD's products."
        ),
        "missingEvidence": (
            "Competitor revenue performance and segment sales breakdowns "
            "to distinguish company-specific from sector-wide decline."
        ),
    },
    {
        "sourceFactIds": ["amd-mi300"],
        "transmission": "SUPPLY", "direction": "POSITIVE",
        "triggerCondition": (
            "If AMD successfully launches and ramps MI300 production "
            "as guided in Q4 2023."
        ),
        "conditionalEffect": (
            "Increased supply of the MI300 accelerator could potentially "
            "stabilize or improve future revenue trends."
        ),
        "counterHypothesis": (
            "The ramp-up of MI300 production faces unforeseen supply chain "
            "bottlenecks or manufacturing delays."
        ),
        "missingEvidence": (
            "Detailed supply chain metrics and inventory levels for MI300 "
            "and specific manufacturing throughput data."
        ),
    },
]


def test_v3_four_distinct_scenarios_are_candidates_not_proven_facts() -> None:
    expected_flags = [
        "IMPLICIT_TRIGGER_CONDITION_SCOPED_BY_EFFECT",
        "IMPLICIT_TRIGGER_CONDITION_SCOPED_BY_EFFECT",
        "DECLARATIVE_COUNTER_AS_UNVERIFIED_ALTERNATIVE",
        "DECLARATIVE_COUNTER_AS_UNVERIFIED_ALTERNATIVE",
    ]
    for proposal, expected in zip(V3, expected_flags, strict=True):
        check = review_scenario(proposal)
        assert check["status"] == "CONDITIONAL_RESEARCH_CANDIDATE"
        assert not check["blockers"]
        assert expected in check["warnings"]
        assert check["allStatementsAreAssumptions"] is True
        assert check["causalityProven"] is False
        assert check["semanticEntailmentVerified"] is False


def test_actual_case_audit_recovers_four_old_structural_rejections_no_model() -> None:
    cohort = json.loads(COHORT.read_text(encoding="utf-8"))
    original = official.run_cohort(cohort, SourceFactProvider())
    for case, proposals in zip(
        original["cases"], [[V3[0]], [V3[1]], V3[2:]], strict=True,
    ):
        case["economicMechanisms"] = []
        case["rejectedEconomicMechanisms"] = [
            {
                "reason": (
                    "TRIGGER_NOT_CONDITIONAL"
                    if proposal in V3[:2] else "COUNTER_HYPOTHESIS_NOT_CONDITIONAL"
                ),
                "proposal": proposal,
            } for proposal in proposals
        ]
    saved = copy.deepcopy(original)
    audited = official.audit_existing(cohort, original)
    assert original == saved
    assert audited["auditModelCalls"] == 0
    assert audited["recoveredCandidateCount"] == 4
    assert sum(
        len(row["economicMechanisms"]) for row in audited["cases"]
    ) == 0
    assert [
        len(row["recoveredResearchCandidates"]) for row in audited["cases"]
    ] == [1, 1, 2]
    assert audited["crossEventReview"]["recoveredCandidateCounts"] == {
        "MSFT-2023-07-25-Q4": 1,
        "GOOGL-2023-07-25-Q2": 1,
        "AMD-2023-08-01-Q2": 2,
    }
    for case in audited["cases"]:
        for restored in case["recoveredResearchCandidates"]:
            assert restored["recovery"] == (
                "STRUCTURAL_REASSESSMENT_NOT_MODEL_REGENERATION"
            )
            assert restored["causalityProven"] is False
            assert restored["humanReviewRequired"] is True


def test_unsupported_numeric_or_regulation_channel_not_recovered() -> None:
    cohort = json.loads(COHORT.read_text(encoding="utf-8"))
    old = official.run_cohort(cohort, SourceFactProvider())
    malformed = {
        **V3[0],
        "transmission": "REGULATION",
        "triggerCondition": "If sales exceed 999% next year",
    }
    old["cases"][0]["rejectedEconomicMechanisms"] = [
        {"reason": "TRIGGER_NOT_CONDITIONAL", "proposal": malformed},
    ]
    audited = official.audit_existing(cohort, old)
    assert not audited["cases"][0]["recoveredResearchCandidates"]


def test_unconditional_effect_and_actual_price_return_remain_blocked() -> None:
    p = {**V3[0], "conditionalEffect": "Microsoft generated higher profits."}
    assert review_scenario(p)["blockers"]
    p = {
        **V3[0],
        "conditionalEffect": "Shares fell by 3.1% after the release.",
    }
    assert "OBSERVED_MARKET_RESULT_NOT_SUPPORTED" in review_scenario(p)["blockers"]
