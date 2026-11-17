"""Source-faithful atomic claim checks, guarded recovery and adversarial inputs."""
from __future__ import annotations

import json

from test_historical_news_impact_test import answer, sample_case

from tris_market_event_intelligence import historical_news_impact_test_cli as cli
from tris_market_event_intelligence.historical_claim_verification import (
    CausalClaim,
    EvidenceClaim,
    verify_claims,
)


def _causal() -> CausalClaim:
    return CausalClaim(
        triggerCondition="If the production delay persists through the quarter",
        transmissionChannel="REVENUE",
        potentialEffect="Revenue could weaken as shipments may be delayed.",
        counterHypothesis="If production recovers promptly, sales might remain stable.",
    )


def _quote_case() -> tuple[cli.CaseInput, list[dict], dict[str, dict]]:
    case = cli.CaseInput.model_validate(sample_case())
    quotes, lookup = cli._quote_anchors(case)
    return case, quotes, lookup


def test_valid_exact_quote_and_conditional_mechanism_not_causal_proof() -> None:
    _, _, lookup = _quote_case()
    a1 = "a1:q0"
    claim = EvidenceClaim(
        quoteId=a1, sourceExcerpt=lookup[a1]["text"],
        epistemicType="REPORTED_FACT",
    )
    check = verify_claims(
        evidence=[claim], causal=_causal(), quote_lookup=lookup,
        permitted_quote_ids=[a1], claimed_name="Example Devices",
    )
    assert check["status"] == "SOURCE_QUOTE_MATCHED_CONDITIONAL"
    assert check["evidence"][0]["sourceMatch"] == "EXACT_SUBSTRING"
    assert check["semanticallyVerified"] is False
    assert check["causalityProven"] is False
    assert check["humanReviewRequired"] is True


def test_fabricated_excerpt_cannot_borrow_real_quote_id() -> None:
    _, _, lookup = _quote_case()
    check = verify_claims(
        evidence=[EvidenceClaim(
            quoteId="a1:q0",
            sourceExcerpt="Example Devices confirmed an official government prohibition.",
            epistemicType="REPORTED_FACT",
        )],
        causal=_causal(), quote_lookup=lookup, permitted_quote_ids=["a1:q0"],
    )
    assert check["status"] == "REJECTED_UNSUPPORTED_CLAIM"
    assert any(x["reason"] == "EXCERPT_NOT_IN_SOURCE" for x in check["claimChecks"])


def test_allegation_is_never_upgraded_to_reported_fact() -> None:
    _, _, lookup = _quote_case()
    extra = dict(lookup)
    extra["x:q0"] = {"text": "A source allegedly confirmed espionage.",
                     "quoteId": "x:q0", "articleId": "x"}
    check = verify_claims(
        evidence=[EvidenceClaim(
            quoteId="x:q0",
            sourceExcerpt="A source allegedly confirmed espionage.",
            epistemicType="REPORTED_FACT",
        )],
        causal=_causal(), quote_lookup=extra, permitted_quote_ids=["x:q0"],
    )
    assert any(x["reason"] == "ALLEGATION_AS_FACT" for x in check["claimChecks"])


def test_observed_market_outcome_blocked_even_with_conditional_other_fields() -> None:
    _, _, lookup = _quote_case()
    causal = _causal().model_copy(update={
        "potentialEffect": "The shares dropped 9 percent yesterday following the news.",
    })
    check = verify_claims(
        evidence=[EvidenceClaim(
            quoteId="a1:q0", sourceExcerpt=lookup["a1:q0"]["text"],
            epistemicType="REPORTED_FACT",
        )],
        causal=causal, quote_lookup=lookup, permitted_quote_ids=["a1:q0"],
    )
    assert any(x["reason"] == "UNVERIFIED_MARKET_OUTCOME" for x in check["claimChecks"])


def test_unsupported_entity_and_unconditional_impact_rejected() -> None:
    _, _, lookup = _quote_case()
    check = verify_claims(
        evidence=[EvidenceClaim(
            quoteId="a1:q0", sourceExcerpt=lookup["a1:q0"]["text"],
            epistemicType="REPORTED_FACT",
        )],
        causal=_causal().model_copy(update={
            "potentialEffect": "The firm faces a guaranteed complete revenue decline.",
        }),
        quote_lookup=lookup, permitted_quote_ids=["a1:q0"],
        claimed_name="Invented Holdings",
    )
    assert {x["reason"] for x in check["claimChecks"]} >= {
        "ENTITY_NOT_IN_EVIDENCE", "NONCONDITIONAL_IMPACT",
    }


def test_wrong_quote_id_and_duplicate_evidence_rejected() -> None:
    _, _, lookup = _quote_case()
    claim = EvidenceClaim(
        quoteId="a1:q0", sourceExcerpt=lookup["a1:q0"]["text"],
        epistemicType="REPORTED_FACT",
    )
    check = verify_claims(
        evidence=[claim, claim], causal=_causal(),
        quote_lookup=lookup, permitted_quote_ids=["a1:q0"],
    )
    assert any(x["reason"] == "DUPLICATE_SOURCE_CLAIM" for x in check["claimChecks"])
    check = verify_claims(
        evidence=[claim], causal=_causal(),
        quote_lookup=lookup, permitted_quote_ids=["a2:q0"],
    )
    assert any(x["reason"] == "CLAIM_QUOTE_NOT_ALLOWED" for x in check["claimChecks"])


def _recovery_response(lookup: dict[str, dict]) -> dict:
    causal = _causal().model_dump()
    return {
        "companyHypotheses": [{
            "companyName": "Example Devices",
            "direction": "UNCERTAIN",
            "evidenceQuoteIds": ["a1:q0"],
            "mechanism": "If output remains reduced, the company's sales could weaken.",
            "uncertainty": "The delay might be temporary and sales may be rescheduled.",
            "sourceClaims": [{
                "quoteId": "a1:q0",
                "sourceExcerpt": lookup["a1:q0"]["text"],
                "epistemicType": "REPORTED_FACT",
            }],
            "causalClaim": causal,
        }],
        "sectorHypotheses": [{
            "sector": "SEMICONDUCTORS",
            "market": "US",
            "direction": "UNCERTAIN",
            "intensity": "MODERATE",
            "confidence": "LOW",
            "evidenceQuoteIds": ["a2:q0"],
            "mechanism": "If delays spread to suppliers, sector revenue could soften.",
            "uncertainty": "The scope might remain restricted to the named producer.",
            "sourceClaims": [{
                "quoteId": "a2:q0",
                "sourceExcerpt": lookup["a2:q0"]["text"],
                "epistemicType": "REPORTED_FACT",
            }],
            "causalClaim": causal,
        }],
    }


class FourStageProvider:
    model_id = "offline-four-stage"

    def __init__(self, recovery: dict):
        self.recovery = recovery
        self.calls = 0

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        self.calls += 1
        data = json.loads(user)
        assert "futureReturn" not in data
        if "claims" in schema["properties"]:
            return json.dumps({"claims": [
                {"articleId": "a1", "quoteId": "a1:q0",
                 "kind": "FACT", "stage": "INITIAL_REPORT"},
                {"articleId": "a2", "quoteId": "a2:q0",
                 "kind": "FACT", "stage": "NEW_FACT"},
            ]})
        if "companies" in schema["properties"]:
            return json.dumps({
                "companies": [{"companyName": "Example Devices",
                               "evidenceQuoteIds": ["a1:q0"]}],
                "sectors": [{"sector": "SEMICONDUCTORS",
                             "evidenceQuoteIds": ["a2:q0"]}],
            })
        if "companyHypotheses" in schema["properties"]:
            assert "eventGroups" in data
            assert data["historicalOutcomesAvailable"] is False
            return json.dumps(self.recovery)
        result = answer(cli.CaseInput.model_validate(sample_case()))
        result["impacts"] = []
        return json.dumps(result)


def test_full_four_stage_recovery_keeps_company_unverified_without_ticker() -> None:
    _, _, lookup = _quote_case()
    provider = FourStageProvider(_recovery_response(lookup))
    # A named company without the explicit issuer mapping remains unverified.
    data = sample_case()
    data["knownTickers"] = []
    report = cli.run_case(
        cli.CaseInput.model_validate(data),
        provider,
        use_claim_extraction=True,
        use_target_selection=True,
    )
    assert provider.calls == 4
    assert report["modelCalls"] == 4
    assert report["impactRecoveryUsed"] is True
    assert report["impactRecoveryError"] is None
    assert len(report["companyImpactHypotheses"]) == 1
    company = report["companyImpactHypotheses"][0]
    assert company["ticker"] is None
    assert company["identityVerified"] is False
    assert company["verifiedStockImpact"] is False
    assert company["claimVerification"]["status"] == "SOURCE_QUOTE_MATCHED_CONDITIONAL"
    assert len(report["impacts"]) == 1
    assert report["impacts"][0]["targetType"] == "SECTOR"
    assert report["impacts"][0]["direction"] == "UNCERTAIN"
    assert report["impacts"][0]["intensity"] == "MODERATE"
    assert report["impacts"][0]["claimVerification"]["causalityProven"] is False
    assert report["marketPricesRead"] is False
    assert report["accuracyVerified"] is False


def test_individual_unsupported_recovery_claim_quarantined_while_valid_survives() -> None:
    case, _, lookup = _quote_case()
    recovery = _recovery_response(lookup)
    recovery["companyHypotheses"][0]["sourceClaims"][0]["sourceExcerpt"] = (
        "The president officially banned processor shipments yesterday."
    )
    provider = FourStageProvider(recovery)
    report = cli.run_case(
        case, provider, use_claim_extraction=True, use_target_selection=True,
    )
    assert report["companyImpactHypotheses"] == []
    assert report["rejectedCompanyImpactHypotheses"][0]["reason"] == (
        "CLAIM_VERIFICATION_FAILED"
    )
    assert any(
        c["reason"] == "EXCERPT_NOT_IN_SOURCE"
        for c in report["impactClaimChecks"][0]["verification"]["claimChecks"]
    )
    assert len(report["impacts"]) == 1
    assert report["impacts"][0]["targetType"] == "SECTOR"
    assert report["impactValidationComplete"] is False


def test_no_recovery_call_when_initial_impact_already_accepted() -> None:
    # The existing three-pass behaviour stays unchanged for successful cases.
    from test_historical_impact_target_grounding import (
        test_full_three_stage_workflow_passes_grounded_targets,
    )
    test_full_three_stage_workflow_passes_grounded_targets()

class PrimaryAcceptedProvider(FourStageProvider):
    """Primary sector passes the old syntax gate, but not atomic checking."""

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        if "claims" not in schema["properties"] and (
            "companies" not in schema["properties"]
            and "companyHypotheses" not in schema["properties"]
        ):
            self.calls += 1
            case = cli.CaseInput.model_validate(sample_case())
            result = answer(case)
            result["impacts"] = [result["impacts"][1]]
            result["impacts"][0]["direction"] = "UNCERTAIN"
            result["impacts"][0]["intensity"] = "MODERATE"
            return json.dumps(result)
        return super().generate_json(system=system, user=user, schema=schema)


def test_primary_accepted_impact_still_requires_atomic_verification() -> None:
    case, _, lookup = _quote_case()
    provider = PrimaryAcceptedProvider(_recovery_response(lookup))
    report = cli.run_case(
        case, provider, use_claim_extraction=True, use_target_selection=True,
        verify_all_impacts=True,
    )
    assert provider.calls == 4
    assert report["impactRecoveryUsed"]
    assert report["verifyAllImpacts"]
    assert len(report["initialAcceptedImpacts"]) == 1
    assert "claimVerification" not in report["initialAcceptedImpacts"][0]
    assert len(report["impacts"]) == 1
    assert report["impacts"][0]["sector"] == "SEMICONDUCTORS"
    assert report["impacts"][0]["claimVerification"]["status"] == (
        "SOURCE_QUOTE_MATCHED_CONDITIONAL"
    )
    assert len(report["impactClaimChecks"]) == 2


def test_primary_accepted_impact_rejected_when_atomic_claim_is_fabricated() -> None:
    case, _, lookup = _quote_case()
    recovery = _recovery_response(lookup)
    recovery["companyHypotheses"] = []
    recovery["sectorHypotheses"][0]["sourceClaims"][0]["sourceExcerpt"] = (
        "The market plunged following the official prohibition."
    )
    provider = PrimaryAcceptedProvider(recovery)
    report = cli.run_case(
        case, provider, use_claim_extraction=True, use_target_selection=True,
        verify_all_impacts=True,
    )
    assert provider.calls == 4
    assert report["impacts"] == []
    assert report["initialAcceptedImpacts"]
    assert any(row["reason"] == "NEWS_TEST_CLAIM_VERIFICATION_MISSING"
               for row in report["rejectedImpacts"])
    assert report["impactValidationComplete"] is False
    assert report["impactClaimChecks"][0]["verification"]["status"] == (
        "REJECTED_UNSUPPORTED_CLAIM"
    )
