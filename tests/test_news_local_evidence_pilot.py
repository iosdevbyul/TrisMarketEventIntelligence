"""Real source-pinned NEWS small model pilot, no fabricated human labels."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError
from test_news_materiality_local import _make_client, _model_result
from test_news_research_worker import files

from tris_market_event_intelligence.news_local_evidence_pilot import (
    _failure_code,
    _schema_failure_field,
    audit_local_news_pilot,
)
from tris_market_event_intelligence.news_review_sample import (
    prepare_news_review_sample,
)


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    cohort = files(tmp_path)[0]
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps(prepare_news_review_sample(
        cohort_path=cohort, max_selected=7,
    )), encoding="utf-8")
    return cohort, sample


def test_preview_reads_real_cohort_without_network(tmp_path: Path) -> None:
    cohort, sample = _inputs(tmp_path)
    raw = cohort.read_bytes()
    report = audit_local_news_pilot(
        cohort_path=cohort, sample_path=sample,
    )
    assert report["status"] == "PREVIEW_ONLY"
    assert report["selectedArticles"] == 2
    assert report["totalOriginalArticles"] == 12
    assert report["modelCalls"] == 0
    assert report["items"] == []
    assert report["humanGoldUsed"] is False
    assert report["productionDbModified"] is False
    assert cohort.read_bytes() == raw


def test_real_two_news_contract_validation_no_gold(tmp_path: Path) -> None:
    cohort, sample = _inputs(tmp_path)
    original = json.loads(cohort.read_text())["items"]
    calls = []
    with _make_client(original, calls) as client:
        # Shared test transport contains the installed gemma2:latest fixture.
        report = audit_local_news_pilot(
            cohort_path=cohort, sample_path=sample, model_id="gemma2:latest",
            run=True, limit=2, client=client,
        )
    assert report["status"] == "UNVERIFIED_LOCAL_NEWS_PILOT_RECORDED"
    assert report["modelCalls"] == 2
    assert len(calls) == 3
    assert report["counts"]["ACCEPTED_UNVERIFIED_HYPOTHESIS"] == 2
    assert all(item["hypothesis"]["independent_human_gold"] is False
               for item in report["items"])
    assert report["independentNewsAccuracyMeasured"] is False
    assert report["modelImprovementValidated"] is False
    assert report["completeCohortInferred"] is False
    assert report["stockPricesUsed"] is False


def test_invented_source_entity_is_rejected_without_fake_gold(tmp_path: Path) -> None:
    cohort, sample = _inputs(tmp_path)
    original = json.loads(cohort.read_text())["items"]
    indexed = {row["articleId"]: row for row in original}
    selected_id = json.loads(sample.read_text())["items"][0]["articleId"]
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [
                {"name": "gemma2:latest"},
            ]})
        payload = json.loads(request.content)
        sent = json.loads(payload["messages"][1]["content"])
        article = indexed[sent["article_id"]]
        result = _model_result(article, "CRITICAL")
        if "Major regulatory decision" not in article["title"]:
            result["entity"] = "Routine local community"
        if article["articleId"] == selected_id:
            result["entity"] = "Invented unmentioned corporation"
        return httpx.Response(200, json={"message": {
            "content": json.dumps(result),
        }})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = audit_local_news_pilot(
            cohort_path=cohort, sample_path=sample, model_id="gemma2:latest",
            run=True, limit=2, client=client,
        )
    assert report["counts"]["REJECTED_OR_MODEL_RUNTIME_FAILED"] == 1
    assert report["items"][0]["contractAccepted"] is False
    assert report["items"][0]["failureCode"] == "SOURCE_EVIDENCE_MISMATCH"
    assert report["counts"]["SOURCE_EVIDENCE_MISMATCH"] == 1
    assert report["items"][0]["hypothesis"] is None
    assert report["modelCalls"] == 2


def test_no_local_model_means_no_inference(tmp_path: Path) -> None:
    cohort, sample = _inputs(tmp_path)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={"models": []})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = audit_local_news_pilot(
            cohort_path=cohort, sample_path=sample, run=True, client=client,
        )
    assert report["status"] == "LOCAL_MODEL_NOT_INSTALLED"
    assert calls == ["/api/tags"]
    assert report["modelDownloadedOrInstalled"] is False


def test_modified_source_sample_fails_before_network(tmp_path: Path) -> None:
    cohort, sample = _inputs(tmp_path)
    data = json.loads(sample.read_text())
    data["items"][0]["title"] = "Injected altered headline"
    sample.write_text(json.dumps(data))
    calls = []
    with (_make_client(json.loads(cohort.read_text())["items"], calls) as client,
          pytest.raises(ValueError, match="provenance")):
            audit_local_news_pilot(
                cohort_path=cohort, sample_path=sample,
                model_id="gemma2:latest", run=True, client=client,
            )
    assert calls == []


@pytest.mark.parametrize("model", ["gemma4:latest", "gemma4:31b-cloud", "unknown"])
def test_disallowed_model_rejected_before_io(tmp_path: Path, model: str) -> None:
    with pytest.raises(ValueError, match="approved"):
        audit_local_news_pilot(
            cohort_path=tmp_path / "missing",
            sample_path=tmp_path / "missing2",
            model_id=model,
        )


@pytest.mark.parametrize("limit", [-1, 0, 4, True])
def test_unbounded_limit_rejected(tmp_path: Path, limit) -> None:
    with pytest.raises(ValueError, match="1..3"):
        audit_local_news_pilot(
            cohort_path=tmp_path / "missing",
            sample_path=tmp_path / "missing2", limit=limit,
        )


@pytest.mark.parametrize(("failure", "code"), [
    (httpx.ReadTimeout("private socket text"), "MODEL_TIMEOUT"),
    (httpx.ConnectError("secret internal URL"), "MODEL_TRANSPORT_ERROR"),
    (ValueError("Model JSON exceeds bounded size"), "MODEL_JSON_TOO_LARGE"),
    (ValueError("Model article provenance mismatch"), "ARTICLE_PROVENANCE_MISMATCH"),
    (ValueError("Unassessed source must not assert a direction"),
     "ABSTENTION_DIRECTION_MISMATCH"),
    (ValueError("unexpected secret model output"), "MODEL_CONTENT_INVALID"),
    (TypeError("raw model data must not appear"), "MODEL_RESPONSE_INVALID"),
])
def test_failure_codes_never_expose_raw_exception_text(
    failure: Exception, code: str,
) -> None:
    result = _failure_code(failure)
    assert result == code
    assert "secret" not in result.lower()
    assert "private" not in result.lower()


def test_validation_field_report_uses_allowlist_only() -> None:
    from tris_market_event_intelligence.news_materiality_hypothesis import (
        NewsMaterialityHypothesis,
    )

    with pytest.raises(ValidationError) as failure:
        NewsMaterialityHypothesis.model_validate({
            "schema_version": "news_materiality_hypothesis_v1",
            "article_id": 7,
            "text_sha256": "f" * 64,
            "entity": "Some Entity",
            "event_summary": "Event appeared in the source",
            "fact_status": "REPORTED",
            "potential_magnitude": "HIGH",
            "direction": "UNCERTAIN",
            "evidence_excerpt": "Some Entity reported something",
            "conditional_mechanism": "If verified the organizational impact may be meaningful",
            "verification_needed": [],
            "magnitude_basis": "HUMAN_OR_MODEL_HYPOTHESIS_UNCALIBRATED",
            "market_prices_used": True,
            "price_prediction": False,
            "independent_human_gold": False,
        })
    assert _schema_failure_field(failure.value) == "market_prices_used"


def test_structured_ollama_format_sent_in_pilot(tmp_path: Path) -> None:
    cohort, sample = _inputs(tmp_path)
    articles = json.loads(cohort.read_text())["items"]
    indexed = {row["articleId"]: row for row in articles}
    seen_formats = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "gemma2:latest"}]})
        payload = json.loads(request.content)
        seen_formats.append(payload["format"])
        user = json.loads(payload["messages"][1]["content"])
        article = indexed[user["article_id"]]
        result = _model_result(article, "HIGH")
        if "Major regulatory decision" not in article["title"]:
            result["entity"] = "Routine local community"
        return httpx.Response(200, json={"message": {"content": json.dumps(result)}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = audit_local_news_pilot(
            cohort_path=cohort, sample_path=sample,
            model_id="gemma2:latest", limit=2, run=True, client=client,
        )
    assert report["modelCalls"] == 2
    assert len(seen_formats) == 2
    assert all(isinstance(fmt, dict) for fmt in seen_formats)
    assert all("verification_needed" in fmt["required"] for fmt in seen_formats)
    assert all(fmt["properties"]["price_prediction"]["enum"] == [False]
               for fmt in seen_formats)


def test_schema_rejection_includes_safe_field_name_only(tmp_path: Path) -> None:
    cohort, sample = _inputs(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "gemma2:latest"}]})
        return httpx.Response(200, json={"message": {
            "content": json.dumps({"article_id": 99, "privateNews": "secret"}),
        }})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = audit_local_news_pilot(
            cohort_path=cohort, sample_path=sample,
            model_id="gemma2:latest", limit=1, run=True, client=client,
        )
    assert report["counts"]["MODEL_SCHEMA_INVALID"] == 1
    assert report["items"][0]["schemaFailureField"] in {
        "schema_version", "text_sha256", "entity", "event_summary",
        "fact_status", "potential_magnitude", "direction", "evidence_excerpt",
        "conditional_mechanism", "verification_needed", "magnitude_basis",
    }
    assert "secret" not in json.dumps(report)


@pytest.mark.parametrize(("done_reason", "content", "expected"), [
    ("length", '{"article_id":', "MODEL_GENERATION_TRUNCATED"),
    ("stop", "", "MODEL_EMPTY_CONTENT"),
    ("stop", '{"article_id":', "MODEL_JSON_INVALID"),
])
def test_real_ollama_incomplete_json_is_distinguished_from_schema_failure(
    tmp_path: Path, done_reason: str, content: str, expected: str,
) -> None:
    cohort, sample = _inputs(tmp_path)
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [
                {"name": "gemma2:latest"},
            ]})
        body = json.loads(request.content)
        assert body["format"]["type"] == "object"
        assert body["think"] is False
        assert body["options"]["num_predict"] == 2048
        return httpx.Response(200, json={
            "message": {"content": content},
            "done_reason": done_reason,
        })

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = audit_local_news_pilot(
            cohort_path=cohort, sample_path=sample, model_id="gemma2:latest",
            run=True, limit=1, client=client,
        )
    assert requests == ["/api/tags", "/api/chat"]
    assert report["modelCalls"] == 1
    assert report["items"][0]["failureCode"] == expected
    assert report["counts"][expected] == 1
    assert report["items"][0]["hypothesis"] is None
    assert report["promotionAllowed"] is False
    assert content not in json.dumps(report)
