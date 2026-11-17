"""Three REAL official issuer 2023 releases: source-scoped research regressions."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tris_market_event_intelligence import historical_official_fact_reasoning_cli as official

COHORT = Path("research/independent-historical-news-2023.json")


def _cohort() -> dict:
    return json.loads(COHORT.read_text(encoding="utf-8"))


class SourceFactProvider:
    model_id = "offline-actual-case-research"

    def __init__(self, *, flawed: bool = False):
        self.calls = 0
        self.flawed = flawed

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        self.calls += 1
        packet = json.loads(user)
        assert "factInterpretations" in schema["properties"]
        assert packet["publisherClockVerified"] is False
        assert packet["fullArticleProvided"] is False
        assert packet["pricesAvailable"] is False
        assert packet["outcomesAvailable"] is False
        assert "sourceFactParaphrases" in packet
        facts = packet["sourceFactParaphrases"]
        return json.dumps({
            "factInterpretations": [
                {
                    "factId": fact["factId"],
                    "classification": (
                        "FORWARD_GUIDANCE" if fact["factId"] == "amd-mi300"
                        else "REPORTED_FACT"
                    ),
                    "explanation": (
                        "The issuer's original disclosure reports this "
                        "specific business statement; it is not a price outcome."
                    ),
                    "limitation": (
                        "Publication time and contemporaneous "
                        "market expectations are not independently archived."
                    ),
                } for fact in facts
            ],
            "factRelations": [{
                "fromFactId": facts[0]["factId"],
                "toFactId": facts[1]["factId"],
                "relation": "SAME_DISCLOSURE",
                "explanation": (
                    "These reported facts occur in the same official release."
                ),
            }],
            "mechanisms": [{
                "sourceFactIds": (
                    ["unknown-fact"] if self.flawed
                    else [facts[0]["factId"], facts[1]["factId"]]
                ),
                "transmission": "REVENUE",
                "direction": "UNCERTAIN",
                "triggerCondition": (
                    "If reported sales expectations change after this release"
                ),
                "conditionalEffect": (
                    "Investor assessment of future revenue could adjust "
                    "if the disclosed operating trend persists."
                ),
                "counterHypothesis": (
                    "If the results were expected, valuations might remain stable."
                ),
                "missingEvidence": (
                    "Contemporaneous estimates and independent follow-up "
                    "announcements would be needed to assess surprises."
                ),
            }],
            "overallAssessment": (
                "An issuer disclosure supports reported business facts "
                "but does not prove the causal direction of any stock move."
            ),
        })


def test_real_microsoft_alphabet_amd_sources_reused_without_invented_articles() -> None:
    cohort = _cohort()
    report = official.run_cohort(cohort, SourceFactProvider())
    assert report["modelCalls"] == 3
    assert [r["issuer"] for r in report["cases"]] == [
        "Microsoft", "Alphabet", "AMD",
    ]
    assert sum(len(r["factInterpretations"]) for r in report["cases"]) == 9
    assert sum(len(r["economicMechanisms"]) for r in report["cases"]) == 3
    assert report["crossEventReview"]["caseCount"] == 3
    assert len(report["crossEventReview"]["comparisons"]) == 3
    assert all(p["sourceDocumentsIndependent"]
               for p in report["crossEventReview"]["comparisons"])
    assert all(f["textProvenance"] == "CURATED_PARAPHRASE_NOT_VERBATIM"
               for row in report["cases"] for f in row["factInterpretations"])
    assert all(row["source"]["publisherClockVerified"] is False
               for row in report["cases"])
    assert report["crossEventReview"]["generalizationVerified"] is False
    assert report["crossEventReview"]["pointInTimeReplayVerified"] is False
    assert report["marketPricesRead"] is False
    assert report["trainingPerformed"] is False


def test_amd_forward_guidance_not_mislabeled_as_realized_result() -> None:
    result = official.run_case(_cohort()["cases"][2], SourceFactProvider())
    mi = next(x for x in result["factInterpretations"]
              if x["factId"] == "amd-mi300")
    assert mi["classification"] == "FORWARD_GUIDANCE"
    assert mi["historicalAsOfVerified"] is False


def test_fabricated_fact_reference_is_quarantined_not_promoted() -> None:
    result = official.run_case(
        _cohort()["cases"][1], SourceFactProvider(flawed=True),
    )
    assert result["economicMechanisms"] == []
    assert result["rejectedEconomicMechanisms"][0]["reason"] == (
        "UNSUPPORTED_OR_REJECTED_SOURCE_FACT"
    )
    assert len(result["factInterpretations"]) == 3


def test_future_guidance_classification_conflict_is_not_silently_corrected() -> None:
    case = _cohort()["cases"][2]

    class WrongGuidance(SourceFactProvider):
        def generate_json(self, *, system: str, user: str, schema: dict) -> str:
            response = json.loads(super().generate_json(
                system=system, user=user, schema=schema,
            ))
            response["factInterpretations"][2]["classification"] = "REPORTED_FACT"
            return json.dumps(response)

    result = official.run_case(case, WrongGuidance())
    assert len(result["factInterpretations"]) == 2
    assert result["rejectedFactInterpretations"][0]["reason"] == (
        "FACT_CLASSIFICATION_DISAGREES_WITH_SOURCE"
    )


def test_fake_market_results_cannot_be_smuggled_into_mechanism() -> None:
    class MarketOutcome(SourceFactProvider):
        def generate_json(self, *, system: str, user: str, schema: dict) -> str:
            response = json.loads(super().generate_json(
                system=system, user=user, schema=schema,
            ))
            response["mechanisms"][0]["conditionalEffect"] = (
                "Shares fell 3.1% on Tuesday following the release."
            )
            return json.dumps(response)

    result = official.run_case(_cohort()["cases"][0], MarketOutcome())
    assert result["economicMechanisms"] == []
    assert result["rejectedEconomicMechanisms"][0]["reason"] == (
        "EFFECT_NOT_CONDITIONAL"
    )


def test_bounded_preview_uses_real_cohort_without_model_or_writes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", [
        "historical-official-research", "--cohort", str(COHORT), "--preview",
    ])
    official.main()
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "THREE_ACTUAL_OFFICIAL_CASES_READY"
    assert report["sourceFactCount"] == 9
    assert report["publisherClocksVerified"] is False
    assert report["modelCalls"] == 0


def test_output_cannot_claim_generalization_from_duplicate_releases() -> None:
    report = official.run_cohort(_cohort(), SourceFactProvider())
    cases = report["cases"]
    assert len({x["source"]["sourceId"] for x in cases}) == 3
    with pytest.raises(ValueError, match="THREE_DISTINCT"):
        official.compare_cohort([cases[0], cases[0], cases[1]],
                                cohort_sha=report["sourceCohortSha256"])
