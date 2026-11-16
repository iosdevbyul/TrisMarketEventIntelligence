"""Regression tests for source-grounded target selection and market outcome gates."""
from __future__ import annotations

import json

from tris_market_event_intelligence import historical_news_impact_test_cli as cli
from tris_market_event_intelligence.historical_impact_targets import select_targets
from test_historical_news_impact_test import answer, sample_case


class TargetProvider:
    model_id = "offline-target-proposal"

    def __init__(self, response: dict):
        self.response = response

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        assert "companies" in schema["properties"]
        assert "stockPrices" not in user
        return json.dumps(self.response)


def test_source_quote_grounded_companies_and_sectors_are_separate() -> None:
    case = cli.CaseInput.model_validate(sample_case())
    anchors, _ = cli._quote_anchors(case)
    a1 = next(x["quoteId"] for x in anchors if x["articleId"] == "a1")
    a2 = next(x["quoteId"] for x in anchors if x["articleId"] == "a2")
    targets = select_targets(
        provider=TargetProvider({
            "companies": [
                {"companyName": "Example Devices", "evidenceQuoteIds": [a1]},
                {"companyName": "Invented Corporation", "evidenceQuoteIds": [a2]},
            ],
            "sectors": [
                {"sector": "SEMICONDUCTORS", "evidenceQuoteIds": [a2]},
            ],
        }),
        articles=[a.model_dump() for a in case.articles],
        anchors=anchors, claims=[], as_of=case.asOf,
        known_tickers=[k.model_dump() for k in case.knownTickers],
    )
    assert targets["companies"] == [{
        "companyName": "Example Devices",
        "ticker": "EXDV",
        "tickerProvenance": "USER_SUPPLIED_MAPPING",
        "sector": "SEMICONDUCTORS",
        "evidenceQuoteIds": [a1],
    }]
    assert [x["sector"] for x in targets["sectors"]] == ["SEMICONDUCTORS"]
    assert targets["rejected"][0]["reason"] == "COMPANY_NOT_GROUNDED_IN_SOURCE_QUOTE"


def test_source_named_company_without_mapping_stays_unverified() -> None:
    case = cli.CaseInput.model_validate(sample_case())
    anchors, _ = cli._quote_anchors(case)
    result = select_targets(
        provider=TargetProvider({
            "companies": [{
                "companyName": "Example Devices",
                "evidenceQuoteIds": ["a1:q0"],
            }],
            "sectors": [],
        }),
        articles=[a.model_dump() for a in case.articles],
        anchors=anchors, claims=[], as_of=case.asOf, known_tickers=[],
    )
    assert result["companies"][0]["ticker"] is None
    assert result["companies"][0]["tickerProvenance"] == "UNVERIFIED_COMPANY_CANDIDATE"


def test_invented_observed_market_decline_is_quarantined() -> None:
    case = cli.CaseInput.model_validate(sample_case())
    proposed = answer(case)
    proposed["impacts"][0]["mechanism"] = (
        "The investigation contributed to a broader sell-off of tech stocks "
        "and downward pressure on the S&P 500 yesterday."
    )
    reviewed = cli._validate_answer(
        case, cli.ModelAnswer.model_validate(proposed), cli._quote_anchors(case)[1],
    )
    assert reviewed["impacts"][0]["targetType"] == "SECTOR"
    assert reviewed["rejectedImpacts"][0]["reason"] == (
        "NEWS_TEST_UNSUPPORTED_OBSERVED_MARKET_OUTCOME"
    )
    assert reviewed["impactValidationComplete"] is False
    assert reviewed["rejectedMarketOutcomeClaims"][0]["impactIndex"] == 0


def test_conditional_impact_is_not_mistaken_for_observed_market_result() -> None:
    assert not cli._observed_outcome_claim(
        "If export rules are tightened, technology stocks could decline."
    )
    assert cli._observed_outcome_claim(
        "Market stocks declined sharply after the announcement."
    )
    assert not cli._observed_outcome_claim(
        "The expected future revenue risk could reduce valuation."
    )


def test_unlisted_target_cannot_become_validated_stock() -> None:
    case = cli.CaseInput.model_validate(sample_case())
    original = answer(case)
    targets = {"companies": [], "sectors": [{
        "sector": "SEMICONDUCTORS", "evidenceQuoteIds": ["a2:q0"],
    }], "rejected": []}
    result = cli._validate_answer(
        case, cli.ModelAnswer.model_validate(original),
        cli._quote_anchors(case)[1], targets,
    )
    assert not any(x["targetType"] == "STOCK" for x in result["impacts"])
    assert result["rejectedImpacts"][0]["reason"] == (
        "NEWS_TEST_STOCK_NOT_IN_SOURCE_TARGETS"
    )
    assert result["impacts"][0]["targetType"] == "SECTOR"


def test_narrative_observed_selloff_is_audited() -> None:
    case = cli.CaseInput.model_validate(sample_case())
    original = answer(case)
    original["overallAssessment"] = (
        "The technology stocks sold off after the announcement, causing "
        "the broad market to decline."
    )
    result = cli._validate_answer(
        case, cli.ModelAnswer.model_validate(original),
        cli._quote_anchors(case)[1],
    )
    assert result["rejectedNarrativeClaims"][0]["location"] == "overallAssessment"
    assert not result["impactValidationComplete"]


def test_full_three_stage_workflow_passes_grounded_targets() -> None:
    case = cli.CaseInput.model_validate(sample_case())
    anchors, _ = cli._quote_anchors(case)
    a1 = next(x["quoteId"] for x in anchors if x["articleId"] == "a1")
    a2 = next(x["quoteId"] for x in anchors if x["articleId"] == "a2")

    class ThreePass:
        model_id = "offline-three-stage"
        def __init__(self) -> None:
            self.calls = 0

        def generate_json(self, *, system: str, user: str, schema: dict) -> str:
            self.calls += 1
            request = json.loads(user)
            assert "actualStockReturnPct" not in request
            if "claims" in schema["properties"]:
                return json.dumps({"claims": [
                    {"articleId": "a1", "quoteId": a1,
                     "kind": "FACT", "stage": "INITIAL_REPORT"},
                    {"articleId": "a2", "quoteId": a2,
                     "kind": "FACT", "stage": "NEW_FACT"},
                ]})
            if "companies" in schema["properties"]:
                assert len(request["articleUnderstanding"]) == 2
                return json.dumps({
                    "companies": [{
                        "companyName": "Example Devices", "evidenceQuoteIds": [a1],
                    }],
                    "sectors": [{
                        "sector": "SEMICONDUCTORS", "evidenceQuoteIds": [a2],
                    }],
                })
            assert "targetSelection" in request
            assert len(request["targetSelection"]["companies"]) == 1
            return json.dumps(answer(case))

    provider = ThreePass()
    report = cli.run_case(
        case, provider, use_claim_extraction=True, use_target_selection=True,
    )
    assert provider.calls == 3
    assert report["modelCalls"] == 3
    assert report["impactValidationComplete"]
    assert len(report["impacts"]) == 2
    assert report["marketPricesRead"] is False
    assert report["accuracyVerified"] is False
