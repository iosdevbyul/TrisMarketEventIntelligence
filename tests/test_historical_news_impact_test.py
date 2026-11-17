"""Historical news test: event chaining, stock sectors, signed impacts and blind output."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from tris_market_event_intelligence import historical_news_impact_test_cli as cli


def sample_case() -> dict:
    return {
        "schemaVersion": "historical_news_impact_test_v1",
        "caseId": "sample-2018-001",
        "asOf": "2018-07-12T15:00:00Z",
        "horizonTradingDays": 1,
        "articles": [
            {
                "articleId": "a1",
                "publishedAt": "2018-07-12T11:00:00Z",
                "source": "Example Official Press",
                "title": "Example Devices announces production delay for its flagship processors",
                "body": (
                    "Example Devices reported a production delay affecting its "
                    "flagship processor line. The company said output would "
                    "remain below original plans during the current quarter. "
                    "Its customers may see revised delivery schedules."
                ),
            },
            {
                "articleId": "a2",
                "publishedAt": "2018-07-12T14:30:00Z",
                "source": "Example Business Wire",
                "title": "Example Devices confirms revised processor shipment schedule",
                "body": (
                    "Example Devices confirmed the original production-delay report "
                    "and said shipments may be delayed for several weeks. "
                    "A spokesperson clarified that other chip products "
                    "were not affected by the change."
                ),
            },
        ],
        "knownTickers": [{
            "ticker": "EXDV", "companyName": "Example Devices",
            "sector": "SEMICONDUCTORS",
        }],
    }


def answer(case: cli.CaseInput, *, surprise: bool = False) -> dict:
    _, quotes = cli._quote_anchors(case)
    a1, a2 = next(k for k in quotes if k.startswith("a1:")), next(
        k for k in quotes if k.startswith("a2:")
    )
    result = {
        "eventGroups": [{
            "eventId": "event-1", "articleIds": ["a1", "a2"],
            "summary": "Two reports concern the same processor production disruption.",
        }],
        "eventStagesByArticleId": {
            "a1": "INITIAL_REPORT", "a2": "NEW_FACT",
        },
        "eventLinks": [{
            "earlierArticleId": "a1", "laterArticleId": "a2",
            "relation": "FOLLOW_UP",
            "explanation": "The later update confirms and refines the earlier disruption.",
        }],
        "impacts": [
            {
                "targetType": "STOCK", "ticker": "EXDV",
                "companyName": "Example Devices", "sector": "SEMICONDUCTORS",
                "market": "US", "exposure": "DIRECT", "direction": "NEGATIVE",
                "intensity": "MODERATE",
                "conditionalMovePctRange": {"minPct": -3.0, "maxPct": -1.0},
                "confidence": "LOW",
                "evidenceQuoteIds": [a1, a2],
                "mechanism": (
                    "If reduced quarterly shipments lower expected revenue and earnings, "
                    "investors might revise the issuer's valuation downward."
                ),
                "uncertainty": (
                    "The reports do not establish whether this impact was already "
                    "anticipated by the market or priced into the shares."
                ),
            },
            {
                "targetType": "SECTOR", "ticker": None, "companyName": None,
                "sector": "SEMICONDUCTORS", "market": "US",
                "exposure": "BROAD_SECTOR", "direction": "UNCERTAIN",
                "intensity": "UNASSESSED", "conditionalMovePctRange": None,
                "confidence": "LOW", "evidenceQuoteIds": [a2],
                "mechanism": (
                    "Sector-wide sentiment could be affected if investors interpret "
                    "one issuer's delay as a possible supply constraint."
                ),
                "uncertainty": (
                    "No source establishes whether other sector issuers are "
                    "affected, so a sector-wide price direction is unresolved."
                ),
            },
        ],
        "overallAssessment": (
            "The reports describe the initial delay and later clarification. "
            "Direction estimates are conditional and lack price calibration."
        ),
    }
    if surprise:
        result["futurePriceResult"] = "+11%"
    return result


class FakeProvider:
    model_id = "offline-test-provider"

    def __init__(self, response: dict):
        self.response = response
        self.calls = 0
        self.request: dict | None = None

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        self.calls += 1
        self.request = json.loads(user)
        assert schema["properties"]["impacts"]
        assert "knownOutcomes" not in self.request
        return json.dumps(self.response)


def test_full_multi_news_story_and_signed_impacts() -> None:
    case = cli.CaseInput.model_validate(sample_case())
    provider = FakeProvider(answer(case))
    report = cli.run_case(case, provider)
    assert provider.calls == 1
    assert len(report["eventGroups"]) == 1
    assert len(report["eventLinks"]) == 1
    assert len(report["eventStages"]) == 2
    assert len(report["impacts"]) == 2
    stock, sector = report["impacts"]
    assert stock["ticker"] == "EXDV"
    assert stock["tickerProvenance"] == "USER_SUPPLIED_MAPPING"
    assert stock["direction"] == "NEGATIVE"
    assert stock["conditionalMovePctRange"] == {"minPct": -3.0, "maxPct": -1.0}
    assert len(stock["evidence"]) == 2
    assert sector["direction"] == "UNCERTAIN"
    assert sector["conditionalMovePctRange"] is None
    assert report["status"] == "BLIND_NEWS_IMPACTS_RECORDED_UNCALIBRATED"
    assert report["accuracyVerified"] is False
    assert report["marketPricesRead"] is False
    assert report["outcomesRead"] is False
    assert report["calibrationVerified"] is False
    assert report["trainingReady"] is False
    assert "marketPrice" not in json.dumps(provider.request)
    assert report["newsInputSha256"] == cli.prepare_case(case)["newsInputSha256"]


def test_reject_outcome_data_in_source_contract() -> None:
    supplied = sample_case()
    supplied["actualStockReturnPct"] = -7.5
    with pytest.raises(ValidationError):
        cli.CaseInput.model_validate(supplied)


@pytest.mark.parametrize("mutation", [
    lambda case: case["articles"][1].update(
        {"publishedAt": "2018-07-13T14:00:00Z"}
    ),
    lambda case: case["articles"][1].update(
        {"publishedAt": "2018-07-12T14:30:00"}
    ),
    lambda case: case["articles"][1].update({"articleId": "a1"}),
    lambda case: case.update({"horizonTradingDays": 99}),
    lambda case: case.update({"knownTickers": [
        {"ticker": "INVALID!!!!!!", "companyName": "Example Devices",
         "sector": "SEMICONDUCTORS"}
    ]}),
])
def test_invalid_timing_and_identity_rejected(mutation: object) -> None:
    supplied = sample_case()
    mutation(supplied)
    with pytest.raises((ValueError, ValidationError)):
        cli.CaseInput.model_validate(supplied)


@pytest.mark.parametrize("mutate", [
    lambda result: result["eventGroups"][0].update({"articleIds": ["a1"]}),
    lambda result: result["impacts"][0].update({"evidenceQuoteIds": ["bad:q0"]}),
    lambda result: result["impacts"][0].update({"ticker": "FAKE"}),
    lambda result: result["impacts"][0].update({"companyName": "Other Company"}),
    lambda result: result["impacts"][0].update({
        "conditionalMovePctRange": {"minPct": 0.5, "maxPct": 2.0},
    }),
    lambda result: result["impacts"][0].update({
        "conditionalMovePctRange": {"minPct": -25, "maxPct": -1},
    }),
    lambda result: result["impacts"][1].update({
        "conditionalMovePctRange": {"minPct": 1, "maxPct": 2},
    }),
])
def test_forged_chains_targets_quotes_and_influence_fail_closed(mutate: object) -> None:
    case = cli.CaseInput.model_validate(sample_case())
    result = answer(case)
    mutate(result)
    with pytest.raises((ValueError, ValidationError)):
        cli.run_case(case, FakeProvider(result))


def test_extra_model_claims_are_not_silently_discarded() -> None:
    case = cli.CaseInput.model_validate(sample_case())
    with pytest.raises(ValidationError):
        cli.run_case(case, FakeProvider(answer(case, surprise=True)))


def test_independent_events_do_not_need_invented_relationship() -> None:
    case = cli.CaseInput.model_validate(sample_case())
    result = answer(case)
    result["eventGroups"] = [
        {"eventId": "first", "articleIds": ["a1"], "summary": "First separate reported event."},
        {"eventId": "second", "articleIds": ["a2"], "summary": "Second separate reported event."},
    ]
    result["eventLinks"] = []
    report = cli.run_case(case, FakeProvider(result))
    assert len(report["eventLinks"]) == 0
    assert len(report["eventGroups"]) == 2


def test_abstain_without_hallucinated_prices_or_sector_impact() -> None:
    case = cli.CaseInput.model_validate(sample_case())
    result = answer(case)
    result["impacts"] = []
    report = cli.run_case(case, FakeProvider(result))
    assert report["status"] == "BLIND_NEWS_ABSTAINED"
    assert report["impacts"] == []


def test_prepare_does_not_call_provider_or_persist_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(sample_case()), encoding="utf-8")
    target = tmp_path / "answer.json"
    monkeypatch.setattr(cli, "build_runtime", lambda *a, **k: pytest.fail("model used"))
    monkeypatch.setattr(sys, "argv", [
        "historical-news-impact", "--input", str(path), "--prepare",
    ])
    cli.main()
    preview = json.loads(capsys.readouterr().out)
    assert preview["articleCount"] == 2
    assert preview["quoteAnchorCount"] >= 2
    assert preview["modelCalls"] == 0
    assert not target.exists()


def test_cli_run_writes_private_create_only_sealed_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    supplied = sample_case()
    case = cli.CaseInput.model_validate(supplied)
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(supplied), encoding="utf-8")
    target = tmp_path / "result.json"
    model = FakeProvider(answer(case))
    monkeypatch.setattr(cli, "build_runtime", lambda *a, **k: model)
    monkeypatch.setattr(sys, "argv", [
        "historical-news-impact", "--input", str(input_path),
        "--run", "--output", str(target),
    ])
    cli.main()
    stdout = json.loads(capsys.readouterr().out)
    persisted = json.loads(target.read_text(encoding="utf-8"))
    assert stdout["stockImpacts"] == 1
    assert stdout["sectorImpacts"] == 1
    assert persisted["marketPricesRead"] is False
    assert persisted["newsInputSha256"] == cli.prepare_case(case)["newsInputSha256"]
    assert target.stat().st_mode & 0o777 == 0o600
    assert model.calls == 1
    monkeypatch.setattr(sys, "argv", [
        "historical-news-impact", "--input", str(input_path),
        "--run", "--output", str(target),
    ])
    with pytest.raises(SystemExit) as err:
        cli.main()
    assert err.value.code == 2
    assert model.calls == 1
    assert "NEWS_TEST_OUTPUT_NOT_CREATE_ONLY" in capsys.readouterr().err


def test_source_ticker_proposal_marked_unverified_if_absent() -> None:
    case_dict = sample_case()
    case_dict["knownTickers"] = []
    case = cli.CaseInput.model_validate(case_dict)
    report = cli.run_case(case, FakeProvider(answer(case)))
    stock = report["impacts"][0]
    assert stock["tickerProvenance"] == "UNVERIFIED_MODEL_PROPOSAL"

def test_reversed_or_fabricated_event_links_are_quarantined_not_fatal() -> None:
    case = cli.CaseInput.model_validate(sample_case())
    for change, expected in (
        ({"earlierArticleId": "a2", "laterArticleId": "a1"}, "REVERSED_CHRONOLOGY"),
        ({"laterArticleId": "made-up"}, "UNKNOWN_ARTICLE"),
    ):
        result = answer(case)
        result["eventLinks"][0].update(change)
        report = cli.run_case(case, FakeProvider(result))
        assert report["status"] == "BLIND_NEWS_IMPACTS_RECORDED_UNCALIBRATED"
        assert len(report["eventGroups"]) == 1
        assert report["eventLinks"] == []
        assert report["rejectedEventLinks"][0]["reason"] == expected
        assert len(report["impacts"]) == 2


def test_current_revised_originals_are_explicitly_ineligible_for_historical_accuracy() -> None:
    supplied = sample_case()
    supplied["articles"][0]["sourceIntegrity"] = "CURRENT_PAGE_REVISED_AFTER_CUTOFF"
    supplied["articles"][0]["sourceModifiedAt"] = "2018-07-13T16:00:00+00:00"
    case = cli.CaseInput.model_validate(supplied)
    provider = FakeProvider(answer(case))
    report = cli.run_case(case, provider)
    assert report["revisedSourceArticleIds"] == ["a1"]
    assert report["sourceIntegrityStatus"] == "EXPLORATORY_POST_CUTOFF_REVISED_SOURCE"
    assert report["historicalBenchmarkEligible"] is False
    assert report["accuracyVerified"] is False
    assert report["causalityProven"] is False
    assert "sourceModifiedAt" not in provider.request["articles"][0]
    assert "sourceIntegrity" not in provider.request["articles"][0]


def test_skipped_sources_are_audited_and_ineligible_for_market_benchmark() -> None:
    supplied = sample_case()
    supplied["skippedSources"] = [{
        "articleIndex": 3,
        "host": "www.axios.com",
        "reason": "NEWS_URL_FETCH_HTTP_FAILURE",
    }]
    case = cli.CaseInput.model_validate(supplied)
    provider = FakeProvider(answer(case))
    result = cli.run_case(case, provider)
    assert result["skippedSources"] == supplied["skippedSources"]
    assert result["historicalBenchmarkEligible"] is False
    assert "skippedSources" not in provider.request
    assert result["marketPricesRead"] is False


def test_cli_reports_pydantic_source_field_and_stage_not_article_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    supplied = sample_case()
    supplied["articles"][0]["body"] = "SECRET_NEWS_SOURCE" * 700
    monkeypatch.setattr(cli, "collect_urls", lambda *a, **kw: supplied)
    monkeypatch.setattr(sys, "argv", [
        "historical-news-impact", "--url", "https://techcrunch.com/a",
        "--url", "https://www.axios.com/b",
        "--case-id", "test", "--as-of", "2018-07-12T15:00:00Z",
        "--prepare",
    ])
    with pytest.raises(SystemExit) as err:
        cli.main()
    assert err.value.code == 2
    log = capsys.readouterr().err
    assert "STAGE=SOURCE_INPUT_VALIDATION" in log
    assert "NEWS_TEST_SCHEMA_VALIDATION_FAILED" in log
    assert "articles.0.body" in log
    assert "SECRET_NEWS_SOURCE" not in log


def test_cli_reports_model_schema_failure_and_persists_source_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    case = cli.CaseInput.model_validate(sample_case())
    class BrokenProvider:
        model_id = "broken-provider"
        def generate_json(self, **kwargs: object) -> str:
            return '{"eventGroups": "invalid value"}'

    monkeypatch.setattr(cli, "build_runtime", lambda *a, **kw: BrokenProvider())
    source = tmp_path / "source.json"
    output = tmp_path / "answer.json"
    monkeypatch.setattr(cli, "collect_urls", lambda *a, **kw: case.model_dump())
    monkeypatch.setattr(sys, "argv", [
        "historical-news-impact", "--url", "https://techcrunch.com/a",
        "--url", "https://www.axios.com/b",
        "--case-id", "test", "--as-of", "2018-07-12T15:00:00Z",
        "--run", "--source-output", str(source), "--output", str(output),
    ])
    with pytest.raises(SystemExit) as err:
        cli.main()
    assert err.value.code == 2
    log = capsys.readouterr().err
    assert "STAGE=MODEL_AND_ANSWER_VALIDATION" in log
    assert "NEWS_TEST_SCHEMA_VALIDATION_FAILED" in log
    assert "eventGroups" in log
    assert source.is_file()
    assert not output.exists()
    assert json.loads(source.read_text())["schemaVersion"] == case.schemaVersion


class RangeRetryProvider:
    model_id = "range-retry-fake"

    def __init__(self, responses: list[dict]):
        self.responses = responses
        self.calls = 0
        self.requests: list[dict] = []
        self.schemas: list[dict] = []

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        self.requests.append(json.loads(user))
        self.schemas.append(schema)
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return json.dumps(response)


def test_invalid_conditional_range_triggers_single_model_authored_regeneration() -> None:
    case = cli.CaseInput.model_validate(sample_case())
    malformed = answer(case)
    malformed["impacts"][0]["conditionalMovePctRange"] = {
        "minPct": -15, "maxPct": -1,
    }
    corrected = answer(case)
    provider = RangeRetryProvider([malformed, corrected])
    result = cli.run_case(case, provider)
    assert provider.calls == 2
    assert result["modelCalls"] == 2
    assert result["rangeSchemaRetryUsed"] is True
    assert result["impacts"][0]["conditionalMovePctRange"] == {
        "minPct": -3.0, "maxPct": -1.0,
    }
    assert "correctionContext" not in provider.requests[0]
    assert "correctionContext" in provider.requests[1]
    assert provider.requests[1]["historicalOutcomesAvailable"] is False
    assert "marketPrice" not in json.dumps(provider.requests[1])


def test_persistently_invalid_range_fails_after_one_bounded_retry() -> None:
    case = cli.CaseInput.model_validate(sample_case())
    malformed = answer(case)
    malformed["impacts"][0]["conditionalMovePctRange"] = {
        "minPct": 4.0, "maxPct": -2.0,
    }
    provider = RangeRetryProvider([malformed])
    with pytest.raises(ValidationError):
        cli.run_case(case, provider)
    assert provider.calls == 2


def test_other_model_schema_errors_get_only_one_bounded_regeneration() -> None:
    case = cli.CaseInput.model_validate(sample_case())
    malformed = answer(case)
    malformed["eventGroups"] = "invalid"
    provider = RangeRetryProvider([malformed])
    with pytest.raises(ValidationError):
        cli.run_case(case, provider)
    assert provider.calls == 2


def test_valid_range_retains_single_model_call() -> None:
    case = cli.CaseInput.model_validate(sample_case())
    provider = RangeRetryProvider([answer(case)])
    report = cli.run_case(case, provider)
    assert provider.calls == 1
    assert report["rangeSchemaRetryUsed"] is False
    assert report["modelCalls"] == 1


def test_unknown_article_stage_recovers_only_through_model_regeneration() -> None:
    case = cli.CaseInput.model_validate(sample_case())
    duplicate = answer(case)
    duplicate["eventStagesByArticleId"]["made-up"] = "REACTION"
    regenerated = answer(case)
    provider = RangeRetryProvider([duplicate, regenerated])
    result = cli.run_case(case, provider)
    assert provider.calls == 2
    assert result["modelContractRetryReason"] == "NEWS_TEST_STAGE_COVERAGE_INVALID"
    assert result["rangeSchemaRetryUsed"] is False
    assert result["modelCalls"] == 2
    assert result["eventStages"] == [
        {"articleId": "a1", "stage": "INITIAL_REPORT"},
        {"articleId": "a2", "stage": "NEW_FACT"},
    ]
    correction = provider.requests[1]["correctionContext"]
    assert correction["validArticleIds"] == ["a1", "a2"]
    assert correction["problem"] == {
        "validationCode": "NEWS_TEST_STAGE_COVERAGE_INVALID",
    }
    assert provider.requests[0]["articles"] == provider.requests[1]["articles"]
    assert provider.requests[0]["sourceQuoteAnchors"] == provider.requests[1]["sourceQuoteAnchors"]
    assert "futureStockReturn" not in json.dumps(provider.requests[1])


def test_persistent_unknown_stage_is_rejected_after_two_calls() -> None:
    case = cli.CaseInput.model_validate(sample_case())
    duplicate = answer(case)
    duplicate["eventStagesByArticleId"]["made-up"] = "REACTION"
    provider = RangeRetryProvider([duplicate])
    with pytest.raises(ValueError, match="NEWS_TEST_STAGE_COVERAGE_INVALID"):
        cli.run_case(case, provider)
    assert provider.calls == 2


def test_invalid_quote_relations_regenerate_not_auto_rewritten() -> None:
    case = cli.CaseInput.model_validate(sample_case())
    malformed = answer(case)
    malformed["impacts"][0]["evidenceQuoteIds"] = ["invented:q0"]
    valid = answer(case)
    provider = RangeRetryProvider([malformed, valid])
    result = cli.run_case(case, provider)
    assert provider.calls == 2
    assert result["modelContractRetryReason"] == "NEWS_TEST_QUOTE_NOT_IN_SOURCE"
    assert result["impacts"][0]["evidenceQuoteIds"] == valid["impacts"][0]["evidenceQuoteIds"]


def test_news_with_unsupported_impact_targets_cannot_be_silently_relabelled() -> None:
    case = cli.CaseInput.model_validate(sample_case())
    malformed = answer(case)
    malformed["impacts"][0]["ticker"] = "HALLUC"
    provider = RangeRetryProvider([malformed])
    with pytest.raises(ValueError, match="NEWS_TEST_TICKER_MAPPING_CONFLICT"):
        cli.run_case(case, provider)
    assert provider.calls == 2


def test_stage_schema_requires_exact_source_article_ids() -> None:
    case = cli.CaseInput.model_validate(sample_case())
    provider = RangeRetryProvider([answer(case)])
    cli.run_case(case, provider)
    schema = provider.schemas[0]["properties"]["eventStagesByArticleId"]
    assert schema["type"] == "object"
    assert schema["required"] == ["a1", "a2"]
    assert set(schema["properties"]) == {"a1", "a2"}
    assert schema["additionalProperties"] is False


def test_missing_stage_not_filled_or_invented_by_python() -> None:
    case = cli.CaseInput.model_validate(sample_case())
    missing = answer(case)
    del missing["eventStagesByArticleId"]["a2"]
    provider = RangeRetryProvider([missing])
    with pytest.raises(ValidationError) as exc:
        cli.run_case(case, provider)
    assert any(
        issue["loc"] == ("eventStagesByArticleId",)
        and issue["type"] == "too_short"
        for issue in exc.value.errors()
    )
    assert provider.calls == 2
