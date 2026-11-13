import json
from pathlib import Path

import httpx
import pytest

from tris_market_event_intelligence.collector import NewsStore, parse_rss
from tris_market_event_intelligence.event_store import EventProcessor
from tris_market_event_intelligence.impact import (
    EventInput,
    Impact,
    ImpactResult,
    OpenAICompatibleProvider,
    validate_result,
)
from tris_market_event_intelligence.impact_store import ImpactProcessor

EVIDENCE = "Federal funds rate The Committee raised the interest rate by 25 basis points."


class StubProvider:
    model_id = "offline-test"

    def __init__(self, fail: bool = False, empty: bool = False) -> None:
        self.fail = fail
        self.empty = empty
        self.calls = 0

    def analyze(self, event: EventInput) -> ImpactResult:
        self.calls += 1
        if self.fail:
            raise ConnectionError("network unavailable")
        if self.empty:
            return ImpactResult(impacts=[])
        return ImpactResult(impacts=[
            Impact(
                market="US", sector="TECH", direction="UNCERTAIN",
                horizon="SHORT_TERM",
                rationale="Higher rates may pressure growth valuations, but surprise is unknown.",
                evidence_excerpt="raised the interest rate by 25 basis points",
                uncertainty="HIGH", drivers=["discount rates", "unknown expectations"],
            )
        ])


def setup_db(path: Path) -> tuple[int, int]:
    store = NewsStore(path)
    store.initialize()
    payload = b"""<rss><channel><item><title>Federal funds rate</title>
    <link>https://www.federalreserve.gov/a</link><guid>a1</guid>
    <description>The Committee raised the interest rate by 25 basis points.</description>
    </item><item><title>Federal funds rate</title>
    <link>https://www.federalreserve.gov/b</link><guid>b1</guid>
    <description>The Committee might raise the rate by 25 basis points.</description>
    </item></channel></rss>"""
    rows = parse_rss("fed_monetary", payload, "2026-11-14T00:00:00+00:00")
    store.save("fed_monetary", "https://www.federalreserve.gov/feeds/press_monetary.xml",
               payload, rows, "2026-11-14T00:00:00+00:00", None, None)
    ep = EventProcessor(path)
    assert ep.process_pending().events == 2
    events = ep.list_events()
    assert events[0]["status"] == "VERIFIED"
    assert events[1]["status"] == "CANDIDATE"
    return int(events[0]["id"]), int(events[1]["id"])


def test_provider_contract_and_no_claim_without_evidence() -> None:
    event = EventInput(1, "MONETARY_POLICY", "VERIFIED", "UP", 25,
                       EVIDENCE, "fed_monetary", "https://example.org", "now")
    bad = ImpactResult(impacts=[
        Impact(
            market="KR", sector="FX", direction="POSITIVE",
            horizon="MEDIUM_TERM",
            rationale="This is unsupported evidence in a hypothetical rationale.",
            evidence_excerpt="a completely invented release",
            uncertainty="HIGH", drivers=["exchange rate"],
        )
    ])
    with pytest.raises(ValueError, match="absent"):
        validate_result(event, bad)
    with pytest.raises(ValueError, match="verified"):
        validate_result(EventInput(1, "MONETARY_POLICY", "CANDIDATE", "UP", 25,
                                  EVIDENCE, "fed_monetary", "https://example.org", "now"),
                        ImpactResult(impacts=[]))


def test_provider_http_validation_and_safe_configuration() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        OpenAICompatibleProvider(
            base_url="http://example.com/v1", api_key="secret", model="demo"
        )
    event = EventInput(1, "MONETARY_POLICY", "VERIFIED", "UP", 25,
                       EVIDENCE, "fed_monetary", "https://example.org", "now")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-secret"
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["messages"][0]["role"] == "system"
        assert "evidence" in body["messages"][1]["content"]
        result = StubProvider().analyze(event)
        return httpx.Response(200, json={"choices": [{
            "message": {"content": result.model_dump_json()}
        }]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(
        base_url="https://example.org/v1", api_key="test-secret",
        model="test-model", client=client
    )
    result = provider.analyze(event)
    assert result.impacts[0].uncertainty == "HIGH"


def test_pipeline_success_idempotency_and_candidate_exclusion(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite3"
    verified_id, _ = setup_db(db)
    provider = StubProvider()
    processor = ImpactProcessor(db, provider)
    assert processor.process_pending().processed == 1
    assert processor.process_pending().processed == 0
    assert provider.calls == 1
    data = processor.list_impacts()
    assert len(data) == 1
    assert data[0]["event_id"] == verified_id
    assert data[0]["uncertainty"] == "HIGH"
    assert data[0]["model_id"] == "offline-test"
    assert json.loads(data[0]["drivers_json"]) == ["discount rates", "unknown expectations"]


def test_failure_does_not_acknowledge_and_retry_succeeds(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite3"
    setup_db(db)
    provider = StubProvider(fail=True)
    processor = ImpactProcessor(db, provider)
    assert processor.process_pending().failed == 1
    assert processor.list_impacts() == []
    with processor.store.store.connect() as conn:
        assert conn.execute("SELECT attempts FROM impact_errors").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM impact_runs").fetchone()[0] == 0
    provider.fail = False
    assert processor.process_pending().processed == 1
    with processor.store.store.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM impact_errors").fetchone()[0] == 0


def test_empty_result_records_no_hypothesis(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite3"
    setup_db(db)
    processor = ImpactProcessor(db, StubProvider(empty=True))
    assert processor.process_pending().processed == 1
    assert processor.list_impacts() == []
    with processor.store.store.connect() as conn:
        assert conn.execute("SELECT outcome FROM impact_runs").fetchone()[0] == "NO_HYPOTHESIS"


def test_invalid_provider_output_remains_retryable(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite3"
    setup_db(db)

    class BadProvider:
        model_id = "invalid-provider"

        def analyze(self, event: EventInput) -> ImpactResult:
            return ImpactResult(impacts=[
                Impact(
                    market="US", sector="TECH", direction="NEGATIVE",
                    horizon="SHORT_TERM", rationale="Unsupported but very specific claim.",
                    evidence_excerpt="stock market fell by 3 percent",
                    uncertainty="LOW", drivers=["invented return"],
                )
            ])

    processor = ImpactProcessor(db, BadProvider())
    assert processor.process_pending().failed == 1
    assert processor.list_impacts() == []
