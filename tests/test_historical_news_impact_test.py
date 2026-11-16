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
        "eventStages": [
            {"articleId": "a1", "stage": "INITIAL_REPORT"},
            {"articleId": "a2", "stage": "NEW_FACT"},
        ],
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
