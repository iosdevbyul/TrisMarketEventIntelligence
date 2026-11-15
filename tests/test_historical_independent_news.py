"""Independent news-only case replay; no price, automatic Gold, or Ollama needed."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tris_market_event_intelligence import historical_independent_news_cli as independent

CASE_FILE = Path("research/independent-historical-news-2023.json")


def _cohort() -> dict:
    return json.loads(CASE_FILE.read_text(encoding="utf-8"))


class FakeProvider:
    model_id = "gemma4:e2b"

    def __init__(self) -> None:
        self.calls = 0

    def generate_json(self, *, system: str, user: str, schema: dict | None) -> str:
        assert schema is not None
        assert "not stock prices" in system
        self.calls += 1
        packet = json.loads(user)
        source_id = packet["primaryDocument"]["sourceId"]
        assert packet["fullArticlesAvailable"] is False
        assert packet["stockPriceOrCausalLabelsAvailable"] is False
        return json.dumps({
            "case_id": packet["caseId"],
            "fact_assessments": [
                {
                    "fact_id": item["factId"], "source_id": source_id,
                    "interpretation": "The documented disclosure supplies this fact.",
                }
                for item in packet["sourceFacts"]
            ],
            "hypotheses": [
                {
                    "category": "ISSUER_FUNDAMENTALS",
                    "source_ids": [source_id],
                    "mechanism": "The issuer's information might alter cash flow expectations.",
                    "limitation": "A documentary fact is not causal evidence of a stock move.",
                },
                {
                    "category": "SECTOR_ALTERNATIVE",
                    "source_ids": [],
                    "mechanism": "Sector-wide developments might affect competitive demand.",
                    "limitation": "No independent sector documentation was supplied here.",
                },
                {
                    "category": "MISSING_CONTEXT",
                    "source_ids": [],
                    "mechanism": "Other macro indicators might influence broader expectations.",
                    "limitation": "No timestamped macro release or price context was supplied.",
                },
            ],
            "research_summary": "There are multiple possible mechanisms with missing context.",
            "causal_proof": False,
            "prospective_prediction": False,
        })


def test_three_verified_historical_cases_have_disjoint_primary_sources() -> None:
    cohort = _cohort()
    independent.validate_cohort(cohort)
    assert {x["ticker"] for x in cohort["cases"]} == {"MSFT", "GOOGL", "AMD"}
    assert len({x["announcementDate"] for x in cohort["cases"]}) == 2
    assert sum(len(x["facts"]) for x in cohort["cases"]) == 9
    assert all(x["source"]["publisherClockVerified"] is False for x in cohort["cases"])


def test_independent_inference_and_structural_evaluation_are_separate() -> None:
    cohort = _cohort()
    fake = FakeProvider()
    research = independent.run_cases(cohort, fake)
    evaluation = independent.evaluate_cases(cohort, research)
    assert fake.calls == 3
    assert research["modelCalls"] == 3
    assert research["reviewedLessonsUsed"] is False
    assert evaluation["casesEvaluated"] == 3
    assert evaluation["factAttributionChecksPassed"] == 9
    assert evaluation["heuristicFlagCount"] == 0
    assert evaluation["semanticAccuracyVerified"] is False
    assert evaluation["generalizationImprovementEstablished"] is False
    assert evaluation["independentGoldCreated"] is False


def test_forged_source_or_causality_rejected() -> None:
    case = _cohort()["cases"][0]
    data = json.loads(FakeProvider().generate_json(
        system="not stock prices", user=json.dumps({
            "caseId": case["caseId"], "primaryDocument": case["source"],
            "sourceFacts": case["facts"], "fullArticlesAvailable": False,
            "stockPriceOrCausalLabelsAvailable": False,
        }), schema={},
    ))
    data["fact_assessments"][0]["source_id"] = "invented-article"
    with pytest.raises(ValueError, match="attribution"):
        independent._safe_response(case, json.dumps(data))
    data["fact_assessments"][0]["source_id"] = case["source"]["sourceId"]
    data["causal_proof"] = True
    with pytest.raises(ValueError, match="unsupported"):
        independent._safe_response(case, json.dumps(data))


def test_tampered_case_results_cannot_pass_evaluation() -> None:
    cohort = _cohort()
    research = independent.run_cases(cohort, FakeProvider())
    research["cohortSha256"] = "0" * 64
    with pytest.raises(ValueError, match="integrity mismatch"):
        independent.evaluate_cases(cohort, research)
    research["cohortSha256"] = independent._digest(cohort)
    research["cases"][1]["response"]["hypotheses"][0]["source_ids"] = ["made-up"]
    with pytest.raises(ValueError, match="source"):
        independent.evaluate_cases(cohort, research)


def test_nonnvidia_case_cannot_reuse_same_issuer() -> None:
    cohort = _cohort()
    cohort["cases"][1]["issuer"] = cohort["cases"][0]["issuer"]
    with pytest.raises(ValueError, match="Duplicate"):
        independent.validate_cohort(cohort)


def test_preview_cli_uses_real_sources_without_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    import sys

    monkeypatch.setattr(sys, "argv", [
        "historical-independent", "--cohort", str(CASE_FILE), "--preview",
    ])
    independent.main()
    output = json.loads(capsys.readouterr().out)
    assert output["officialSourceCount"] == 3
    assert output["modelCalls"] == 0
    source = _cohort()
    source_path = tmp_path / "cohort.json"
    source_path.write_text(json.dumps(source))
    fake = independent.run_cases(source, FakeProvider())
    research_file = tmp_path / "research.json"
    research_file.write_text(json.dumps(fake))
    report = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [
        "historical-independent", "--cohort", str(source_path),
        "--evaluate", "--research", str(research_file),
        "--output", str(report),
    ])
    independent.main()
    assert report.stat().st_mode & 0o777 == 0o600
    assert json.loads(report.read_text())["casesEvaluated"] == 3
