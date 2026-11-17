"""Historical price-contaminated quotes must not become as-of evidence."""
from __future__ import annotations

from test_historical_claim_verification import (
    FourStageProvider,
    _recovery_response,
)
from test_historical_news_impact_test import sample_case

from tris_market_event_intelligence import historical_news_impact_test_cli as cli
from tris_market_event_intelligence.historical_claim_verification import (
    CausalClaim,
    EvidenceClaim,
    verify_claims,
)
from tris_market_event_intelligence.historical_evidence_scope import (
    filter_evidence_quotes,
    has_sector_wide_source_evidence,
    is_observed_market_result,
    redact_market_result_sentences,
)


OBSERVED_PRICE = (
    "A drop of 3.1% for Example Devices, previously worth about $5tn, "
    "was the heaviest force pulling the S&P 500 downward recently."
)


def _contaminated_case() -> cli.CaseInput:
    source = sample_case()
    source["articles"][1]["body"] += " " + OBSERVED_PRICE
    return cli.CaseInput.model_validate(source)


def test_real_v13_pattern_is_held_out_but_original_source_is_preserved() -> None:
    case = _contaminated_case()
    original_hash = cli._sha(case.model_dump())
    original, _ = cli._quote_anchors(case)
    safe, rejected = filter_evidence_quotes(original)
    assert is_observed_market_result(OBSERVED_PRICE)
    assert any(row["articleId"] == "a2" for row in rejected)
    assert all("3.1%" not in quote["text"] for quote in safe)
    assert "3.1%" not in redact_market_result_sentences(case.articles[1].body)
    assert "3.1%" in case.articles[1].body
    assert cli._sha(case.model_dump()) == original_hash


def test_price_quote_cannot_pass_atomic_claim_verification_even_if_exact() -> None:
    case = _contaminated_case()
    originals, lookup = cli._quote_anchors(case)
    matched = next(x for x in originals if "3.1%" in x["text"])
    check = verify_claims(
        evidence=[EvidenceClaim(
            quoteId=matched["quoteId"],
            sourceExcerpt=matched["text"],
            epistemicType="REPORTED_FACT",
        )],
        causal=CausalClaim(
            triggerCondition="If shipment constraints persist this quarter",
            transmissionChannel="DEMAND",
            potentialEffect="Demand could weaken until constraints resolve.",
            counterHypothesis="If supply recovers, orders may remain stable.",
        ),
        quote_lookup=lookup,
        permitted_quote_ids=[matched["quoteId"]],
    )
    assert check["status"] == "REJECTED_UNSUPPORTED_CLAIM"
    assert any(x["reason"] == "OBSERVED_MARKET_RESULT_SOURCE_INELIGIBLE"
               for x in check["claimChecks"])


def test_four_model_calls_receive_only_scrubbed_evidence() -> None:
    case = _contaminated_case()
    original_quotes, original_lookup = cli._quote_anchors(case)

    class NoPriceProvider(FourStageProvider):
        def generate_json(self, *, system: str, user: str, schema: dict) -> str:
            assert "3.1%" not in user
            assert "S&P 500 downward" not in user
            return super().generate_json(system=system, user=user, schema=schema)

    provider = NoPriceProvider(_recovery_response(original_lookup))
    result = cli.run_case(
        case, provider, use_claim_extraction=True,
        use_target_selection=True, verify_all_impacts=True,
    )
    assert provider.calls == 4
    assert result["quarantinedMarketOutcomeQuotes"]
    assert all("3.1%" not in quote["text"]
               for quote in filter_evidence_quotes(original_quotes)[0])
    assert result["sourceQuotesExistAtAsOfVerified"] is False
    assert result["historicalBenchmarkEligible"] is None or (
        result["historicalBenchmarkEligible"] is False
    )
    assert result["marketPricesRead"] is False
    assert result["accuracyVerified"] is False


def test_issuer_news_is_not_sector_wide_evidence() -> None:
    assert not has_sector_wide_source_evidence([
        "Nvidia's board launched an investigation after a report about its CEO.",
        "Nvidia confirmed the independent investigation.",
    ])
    assert has_sector_wide_source_evidence([
        "Industry suppliers and chipmakers may face additional costs.",
    ])
