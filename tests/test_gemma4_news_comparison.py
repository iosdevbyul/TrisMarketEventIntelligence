"""Gemma 4 side-by-side pilot: same NEWS, no self-graded model accuracy."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from tris_market_event_intelligence import gemma4_news_comparison
from tris_market_event_intelligence.gemma4_news_comparison import compare_gemma_news
from tris_market_event_intelligence.gemma4_news_comparison_cli import (
    _private_output,
    local_model_preflight,
)
from tris_market_event_intelligence.impact import (
    EventInput,
    Impact,
    ImpactAnalysisError,
    ImpactResult,
)


def event() -> EventInput:
    return EventInput(
        19, "ECONOMIC_INDICATOR", "UNVERIFIED", None, None,
        "Consumer Price Index increased during the last reporting period.",
        "source-news", "https://example.org/article/19",
        "2026-11-15T00:00:00Z", "REPORTED",
    )


def good_result() -> ImpactResult:
    return ImpactResult(impacts=[Impact(
        market="US", sector="FINANCIALS", direction="UNCERTAIN",
        horizon="SHORT_TERM",
        rationale=(
            "If a reported inflation increase shifts future borrowing costs, "
            "bank valuations could change; consensus and verification are unknown."
        ),
        evidence_excerpt="Consumer Price Index increased",
        uncertainty="HIGH", drivers=[], entities=[],
    )])


class FakeAnalyzer:
    def __init__(self, name: str, outcome: str) -> None:
        self.model_id = name
        self.outcome = outcome
        self.calls = []

    def analyze(self, item: EventInput) -> ImpactResult:
        self.calls.append(item)
        if self.outcome == "raise":
            raise ImpactAnalysisError("EVIDENCE_MISMATCH", "Rejected excerpt")
        if self.outcome == "offline":
            raise RuntimeError("Mock transport failure")
        if self.outcome == "empty":
            return ImpactResult(impacts=[])
        return good_result()


def test_exact_same_news_and_evidence_validation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gemma4_news_comparison, "select_news_events",
                        lambda path, **kwargs: [event()])
    original = FakeAnalyzer("gemma2:latest", "empty")
    new = FakeAnalyzer("gemma4:e4b", "valid")
    report = compare_gemma_news(
        db_path=tmp_path / "not-opened.sqlite3", analyzers=(original, new),
        clock=lambda: 1.0,
    )
    assert original.calls == new.calls == [event()]
    assert report["modelCallCount"] == 2
    assert report["items"][0]["modelRuns"][0]["status"] == "ACCEPTED_NO_HYPOTHESIS"
    assert report["items"][0]["modelRuns"][1]["status"] == "ACCEPTED_EVIDENCE_BOUND"
    assert report["candidateImprovementVerified"] is False
    assert report["newsContextUnderstandingVerified"] is False
    assert report["productionDbModified"] is False
    assert report["humanGoldUsed"] is False
    assert report["marketPricesUsed"] is False
    assert "Consumer Price Index" not in json.dumps(report)


def test_preview_never_invokes_models(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gemma4_news_comparison, "select_news_events",
                        lambda path, **kwargs: [event()])
    report = compare_gemma_news(db_path=tmp_path / "db", limit=1)
    assert report["status"] == "PREVIEW_ONLY"
    assert report["modelCallCount"] == 0
    assert report["promotionAllowed"] is False
    assert report["models"] == ["gemma2:latest", "gemma4:e4b"]


def test_runtime_and_validator_rejections_are_sanitized(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gemma4_news_comparison, "select_news_events",
                        lambda path, **kwargs: [event()])
    report = compare_gemma_news(
        db_path=tmp_path / "db",
        analyzers=(FakeAnalyzer("gemma2:latest", "raise"),
                   FakeAnalyzer("gemma4:12b", "offline")),
        models=("gemma2:latest", "gemma4:12b"),
    )
    statuses = [r["status"] for r in report["items"][0]["modelRuns"]]
    assert statuses == ["REJECTED", "MODEL_RUNTIME_FAILED"]
    assert "Mock transport failure" not in json.dumps(report)
    assert report["items"][0]["modelRuns"][0]["failureCode"] == "EVIDENCE_MISMATCH"


@pytest.mark.parametrize("models", [
    ("gemma4:e4b", "gemma2:latest"),
    ("gemma2:latest", "gemma4:latest"),
    ("gemma2:latest", "gemma4:31b-cloud"),
    ("gemma2:latest", "gemma2:latest"),
    ("gemma2:latest", "arbitrary"),
])
def test_ambiguous_or_remote_model_ids_rejected(tmp_path: Path, models) -> None:
    with pytest.raises(ValueError, match="explicitly sized"):
        compare_gemma_news(db_path=tmp_path / "db", models=models)


def test_incorrect_analyzer_model_id_rejected(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gemma4_news_comparison, "select_news_events",
                        lambda path, **kwargs: [event()])
    with pytest.raises(ValueError, match="exact"):
        compare_gemma_news(
            db_path=tmp_path / "db",
            analyzers=(FakeAnalyzer("gemma2:latest", "valid"),
                       FakeAnalyzer("gemma4:12b", "valid")),
        )


def test_local_preflight_no_model_download_or_remote_requests() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url)))
        return httpx.Response(200, json={"models": [
            {"name": "gemma2:latest"}, {"name": "gemma4:e4b"},
        ]})

    with httpx.Client(transport=httpx.MockTransport(handler), trust_env=False) as client:
        response = local_model_preflight(("gemma2:latest", "gemma4:e4b"),
                                         client=client)
    assert response["eachRequestedModelInstalled"] is True
    assert response["modelDownloadedOrInstalled"] is False
    assert calls == [("GET", "http://127.0.0.1:11434/api/tags")]


def test_missing_model_preflight_does_not_claim_runtime_compatibility() -> None:
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={
            "models": [{"name": "gemma2:latest"}],
        })
    )) as client:
        result = local_model_preflight(("gemma2:latest", "gemma4:12b"), client=client)
    assert result["missingModels"] == ["gemma4:12b"]
    assert result["realInferenceCompatibilityVerified"] is False


def test_report_create_only_and_owner_permissions(tmp_path: Path) -> None:
    output = tmp_path / "private.json"
    _private_output(output, {
        "schemaVersion": "gemma4_news_side_by_side_v1", "promotionAllowed": False,
    })
    assert json.loads(output.read_text())["promotionAllowed"] is False
    assert output.stat().st_mode & 0o077 == 0
    with pytest.raises(ValueError, match="Create-only"):
        _private_output(output, {})
