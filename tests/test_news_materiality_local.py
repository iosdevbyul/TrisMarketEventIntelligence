"""News-only local materiality pilot: no price, labels, downloads or auto-run."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from test_news_research_worker import files

from tris_market_event_intelligence.news_materiality_local import (
    generate_materiality_hypotheses,
)


def _model_result(article: dict, level: str) -> dict:
    return {
        "schema_version": "news_materiality_hypothesis_v1",
        "article_id": article["articleId"],
        "text_sha256": article["textSha256"],
        "entity": "Major regulatory decision",
        "event_summary": "A major decision was announced in this fictional news source",
        "fact_status": "REPORTED",
        "potential_magnitude": level,
        "direction": "UNCERTAIN",
        "evidence_excerpt": article["title"],
        "conditional_mechanism": (
            "If the regulatory change takes effect, organizational "
            "compliance processes may require adjustment."
        ),
        "verification_needed": ["Confirm applicable scope"],
        "magnitude_basis": "HUMAN_OR_MODEL_HYPOTHESIS_UNCALIBRATED",
        "market_prices_used": False,
        "price_prediction": False,
        "independent_human_gold": False,
    }


def _make_client(original: list[dict], calls: list, *, bad_article: int = -1):
    indexed = {row["articleId"]: row for row in original}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url)))
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "gemma2:latest"}]})
        assert request.url.path == "/api/chat"
        payload = json.loads(request.content)
        assert payload["model"] == "gemma2:latest"
        assert payload["options"]["temperature"] == 0
        user = json.loads(payload["messages"][1]["content"])
        assert set(user) == {
            "article_id", "text_sha256", "title", "summary", "first_observed_at",
        }
        assert "price" not in payload["messages"][0]["content"].lower() or (
            "never" in payload["messages"][0]["content"].lower()
        )
        row = indexed[user["article_id"]]
        output = _model_result(row, "CRITICAL")
        if row["articleId"] == bad_article:
            output["entity"] = "Invented unmentioned corporation"
        if "Major regulatory decision" not in row["title"]:
            output["entity"] = "Routine local community"
        return httpx.Response(200, json={"message": {
            "content": json.dumps(output),
        }})

    return httpx.Client(transport=httpx.MockTransport(handler), trust_env=False)


def test_preview_never_contacts_local_model(tmp_path: Path):
    paths = files(tmp_path)
    calls = []
    with _make_client(json.loads(paths[0].read_text())["items"], calls) as client:
        preview = generate_materiality_hypotheses(
            cohort_path=paths[0], client=client, limit=12,
        )
    assert preview["status"] == "PREVIEW_ONLY"
    assert preview["modelCalls"] == 0
    assert preview["result"] is None
    assert calls == []


def test_full_local_pilot_returns_pinned_hypotheses_and_no_gold(tmp_path: Path):
    paths = files(tmp_path)
    before = [p.read_bytes() for p in paths[:3]]
    articles = json.loads(paths[0].read_text())["items"]
    calls = []
    with _make_client(articles, calls) as client:
        result = generate_materiality_hypotheses(
            cohort_path=paths[0], client=client, run=True, limit=12,
        )
    assert result["status"] == "RESEARCH_HYPOTHESES_UNVERIFIED"
    assert result["modelCalls"] == 12
    assert len(result["result"]["hypotheses"]) == 12
    assert len(calls) == 13
    assert all("127.0.0.1" in url for _, url in calls)
    assert result["reservedTestGoldScored"] is False
    assert result["modelPerformanceImprovementValidated"] is False
    assert result["promotionAllowed"] is False
    assert result["stockPricesUsed"] is False
    assert not paths[3].exists()
    assert [p.read_bytes() for p in paths[:3]] == before
    assert result["transitions"]
    assert all(x["causalEscalationVerified"] is False
               for x in result["transitions"])


def test_invented_entity_fails_closed_without_partial_model_result(tmp_path: Path):
    paths = files(tmp_path)
    calls = []
    with _make_client(json.loads(paths[0].read_text())["items"], calls,
                      bad_article=1) as client:
        result = generate_materiality_hypotheses(
            cohort_path=paths[0], client=client, run=True, limit=12,
        )
    assert result["status"] == "RESEARCH_INFERENCE_FAILED"
    assert result["result"] is None
    assert result["transitions"] == []
    assert result["promotionAllowed"] is False


def test_missing_installed_model_does_not_attempt_inference(tmp_path: Path):
    paths = files(tmp_path)
    calls = []

    def handler(request: httpx.Request):
        calls.append(request.url.path)
        return httpx.Response(200, json={"models": []})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = generate_materiality_hypotheses(
            cohort_path=paths[0], client=client, run=True, limit=12,
        )
    assert result["status"] == "LOCAL_MODEL_NOT_INSTALLED"
    assert result["modelDownloadedOrInstalled"] is False
    assert calls == ["/api/tags"]


def test_incomplete_cohort_run_is_refused_before_network(tmp_path: Path):
    paths = files(tmp_path)
    calls = []
    with (_make_client(json.loads(paths[0].read_text())["items"], calls) as client,
          pytest.raises(ValueError, match="complete cohort")):
            generate_materiality_hypotheses(
                cohort_path=paths[0], client=client, run=True, limit=11,
            )
    assert calls == []


@pytest.mark.parametrize("model", ["gemma4:latest cloud", "http://host", "", "a" * 101])
def test_untrusted_model_identifiers_rejected(tmp_path: Path, model: str):
    paths = files(tmp_path)
    with pytest.raises(ValueError, match="model name"):
        generate_materiality_hypotheses(cohort_path=paths[0], model_id=model)


def test_source_cohort_hash_integrity_enforced(tmp_path: Path):
    paths = files(tmp_path)
    doc = json.loads(paths[0].read_text())
    doc["items"][0]["title"] = "Tampered headline"
    paths[0].write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="hash mismatch"):
        generate_materiality_hypotheses(cohort_path=paths[0])
