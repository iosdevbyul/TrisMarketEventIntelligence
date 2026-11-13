"""Local LLM contract and multi-event research integration tests."""

import json
import sqlite3
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from tris_market_event_intelligence.api import create_app
from tris_market_event_intelligence.collector import NewsStore, parse_rss
from tris_market_event_intelligence.event_store import EventProcessor
from tris_market_event_intelligence.impact import (
    EventInput,
    Impact,
    ImpactResult,
    OpenAICompatibleProvider,
    is_eligible,
    validate_result,
)
from tris_market_event_intelligence.impact_store import ImpactProcessor


def example_event(status: str = "UNVERIFIED", claim: str = "REPORTED") -> EventInput:
    return EventInput(
        1, "CORPORATE_EVENT", status, None, None,
        "NVIDIA CEO resigns", "bbc_business", "https://example.org/1", "now",
        claim, "THIRD_PARTY_REPORT", ("LEADERSHIP",),
    )


def sample_impact(uncertainty: str = "HIGH", entities: list[str] | None = None) -> Impact:
    return Impact(
        market="US", sector="SEMICONDUCTORS", direction="UNCERTAIN",
        horizon="SHORT_TERM",
        rationale="Executive succession may create uncertainty, conditional on confirmation.",
        evidence_excerpt="NVIDIA CEO resigns", uncertainty=uncertainty,
        drivers=["leadership transition", "report not verified"],
        entities=entities if entities is not None else ["NVIDIA"],
    )


@pytest.mark.parametrize("category", [
    "ECONOMIC_INDICATOR", "EARNINGS", "CORPORATE_EVENT", "REGULATION",
    "GEOPOLITICAL", "SUPPLY_CHAIN", "INDUSTRY", "SERVICE_INCIDENT",
    "MARKET_MOVEMENT",
])
def test_all_general_categories_are_eligible_for_conditional_research(
    category: str,
) -> None:
    item = example_event()
    item = EventInput(
        item.event_id, category, item.status, item.direction, item.change_basis_points,
        item.evidence, item.source, item.source_url, item.detected_at,
        item.claim_kind, item.verification_note, item.tags,
    )
    assert is_eligible(item)
    assert validate_result(item, ImpactResult(impacts=[sample_impact()])).impacts


@pytest.mark.parametrize("claim", ["RUMOR", "FORECAST", "ATTRIBUTED_REPORT"])
def test_unconfirmed_claim_kinds_are_not_analyzed(claim: str) -> None:
    assert not is_eligible(example_event(claim=claim))


def test_unsupported_entity_and_unearned_certainty_are_rejected() -> None:
    with pytest.raises(ValueError, match="absent"):
        validate_result(
            example_event(), ImpactResult(impacts=[
                sample_impact(entities=["A made-up company"])
            ]),
        )
    with pytest.raises(ValueError, match="HIGH"):
        validate_result(
            example_event(), ImpactResult(impacts=[sample_impact(uncertainty="LOW")]),
        )


def test_local_chat_completions_provider_uses_evidence_and_status() -> None:
    event = example_event()

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://127.0.0.1:11434/v1/chat/completions"
        payload = json.loads(request.content)
        message = json.loads(payload["messages"][2]["content"])
        assert message["status"] == "UNVERIFIED"
        assert message["claim_kind"] == "REPORTED"
        assert message["tags"] == ["LEADERSHIP"]
        assert payload["response_format"] == {"type": "json_object"}
        return httpx.Response(
            200, json={"choices": [{"message": {"content": ImpactResult(
                impacts=[sample_impact()]
            ).model_dump_json()}}]},
        )

    provider = OpenAICompatibleProvider(
        base_url="http://127.0.0.1:11434/v1",
        api_key="local-placeholder",
        model="local-mock",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    result = provider.analyze(event)
    assert result.impacts[0].entities == ["NVIDIA"]


def insert_fixture(path: Path) -> None:
    store = NewsStore(path)
    store.initialize()
    payload = b"""<rss><channel>
    <item><title>NVIDIA CEO resigns</title>
    <link>https://example.org/1</link><guid>1</guid></item>
    <item><title>Rumored missile strike</title>
    <link>https://example.org/2</link><guid>2</guid></item>
    <item><title>Consumer Price Index released</title>
    <link>https://example.org/3</link><guid>3</guid></item>
    </channel></rss>"""
    news = parse_rss("bbc_business", payload, "2026-11-14T00:00:00+00:00")
    store.save("bbc_business", "https://example.org/feed", payload, news, "now", None, None)
    assert EventProcessor(path).process_pending().events == 3


class StubLocalModel:
    model_id = "intel-local-fixture"

    def __init__(self) -> None:
        self.calls = 0

    def analyze(self, event: EventInput) -> ImpactResult:
        self.calls += 1
        if event.classification == "CORPORATE_EVENT":
            return ImpactResult(impacts=[sample_impact()])
        return ImpactResult(impacts=[])


def test_multi_category_pipeline_and_api(tmp_path: Path) -> None:
    db = tmp_path / "events.db"
    insert_fixture(db)
    provider = StubLocalModel()
    processor = ImpactProcessor(db, provider)
    first = processor.process_pending()
    assert first.processed == 2
    assert first.impacts == 1
    assert provider.calls == 2
    assert processor.process_pending().processed == 0

    data = processor.list_impacts()
    assert len(data) == 1
    assert json.loads(data[0]["entities_json"]) == ["NVIDIA"]
    response = TestClient(create_app(db)).get("/api/v1/impacts")
    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["eventStatus"] == "UNVERIFIED"
    assert item["claimKind"] == "REPORTED"
    assert item["entities"] == ["NVIDIA"]

    run_response = TestClient(create_app(db)).get("/api/v1/analysis-runs")
    assert [item["outcome"] for item in run_response.json()["items"]] == [
        "COMPLETED", "NO_HYPOTHESIS"
    ]


def test_schema_migrates_legacy_impacts(tmp_path: Path) -> None:
    db = tmp_path / "legacy.db"
    insert_fixture(db)
    processor = ImpactProcessor(db, StubLocalModel())
    processor.initialize()
    with sqlite3.connect(db) as conn:
        conn.execute("ALTER TABLE market_impacts RENAME TO old_impacts")
        conn.execute(
            """CREATE TABLE market_impacts (
            id INTEGER PRIMARY KEY, event_id INTEGER NOT NULL,
            model_id TEXT NOT NULL, prompt_version TEXT NOT NULL,
            market TEXT NOT NULL, sector TEXT NOT NULL, direction TEXT NOT NULL,
            horizon TEXT NOT NULL, rationale TEXT NOT NULL,
            evidence_excerpt TEXT NOT NULL, uncertainty TEXT NOT NULL,
            drivers_json TEXT NOT NULL, generated_at TEXT NOT NULL,
            UNIQUE(event_id, model_id, prompt_version, market, sector, horizon)
            )"""
        )
        conn.execute("DROP TABLE old_impacts")
    processor.initialize()
    with sqlite3.connect(db) as conn:
        fields = [item[1] for item in conn.execute("PRAGMA table_info(market_impacts)")]
    assert "entities_json" in fields
