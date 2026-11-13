"""Safe model rejection categories and persistence."""

import httpx
import pytest

from tris_market_event_intelligence.impact import (
    EventInput,
    Impact,
    ImpactAnalysisError,
    ImpactResult,
    OpenAICompatibleProvider,
    validate_result,
)

EVENT = EventInput(
    6, "ECONOMIC_INDICATOR", "UNVERIFIED", None, None,
    "US inflation rose this month.", "bbc_business", "https://example.org/x",
    "2026-11-14T00:00:00+00:00", "REPORTED",
)


def impact(*, quote: str = "inflation rose",
           entities: list[str] | None = None,
           uncertainty: str = "HIGH") -> Impact:
    return Impact(
        market="US", sector="BROAD_MARKET", direction="UNCERTAIN",
        horizon="SHORT_TERM", rationale="Reported inflation may matter, pending confirmation.",
        evidence_excerpt=quote, uncertainty=uncertainty,
        drivers=["unverified inflation report"], entities=entities or [],
    )


@pytest.mark.parametrize(("item", "code"), [
    (impact(quote="invented text"), "EVIDENCE_MISMATCH"),
    (impact(entities=["Unseen company"]), "ENTITY_MISMATCH"),
    (impact(uncertainty="LOW"), "UNCERTAINTY_MISMATCH"),
])
def test_evidence_checks_expose_safe_codes(item: Impact, code: str) -> None:
    with pytest.raises(ImpactAnalysisError) as error:
        validate_result(EVENT, ImpactResult(impacts=[item]))
    assert error.value.code == code


@pytest.mark.parametrize(("body", "code"), [
    ({"wrong": True}, "PROVIDER_RESPONSE_SHAPE"),
    ({"choices": [{"message": {"content": None}}]}, "PROVIDER_CONTENT_INVALID"),
])
def test_provider_response_codes(body: dict, code: str) -> None:
    client = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=body)
    ))
    model = OpenAICompatibleProvider(
        base_url="http://127.0.0.1:11434/v1",
        api_key="local-placeholder", model="fixture", client=client,
    )
    with pytest.raises(ImpactAnalysisError) as error:
        model.analyze(EVENT)
    assert error.value.code == code


def test_invalid_outer_json_is_categorized() -> None:
    client = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=b"invalid-json")
    ))
    model = OpenAICompatibleProvider(
        base_url="http://127.0.0.1:11434/v1",
        api_key="local-placeholder", model="fixture", client=client,
    )
    with pytest.raises(ImpactAnalysisError) as error:
        model.analyze(EVENT)
    assert error.value.code == "PROVIDER_JSON_INVALID"


def test_failure_saved_without_model_text(tmp_path) -> None:
    from tris_market_event_intelligence.collector import NewsStore, parse_rss
    from tris_market_event_intelligence.event_store import EventProcessor
    from tris_market_event_intelligence.impact_store import ImpactProcessor

    class BadModel:
        model_id = "fixture"

        def analyze(self, event: EventInput) -> ImpactResult:
            return ImpactResult(impacts=[impact(quote="invented secret phrase")])

    db = tmp_path / "news.sqlite3"
    store = NewsStore(db)
    store.initialize()
    raw = b"""<rss><channel><item><guid>1</guid>
    <title>Consumer Price Index released</title>
    <link>https://example.org/1</link></item></channel></rss>"""
    store.save(
        "bbc_business", "https://example.org/feed", raw,
        parse_rss("bbc_business", raw, "now"), "now", None, None,
    )
    EventProcessor(db).process_pending()
    processor = ImpactProcessor(db, BadModel())
    assert processor.process_pending(batch_size=1).failed == 1
    with store.connect() as connection:
        code = connection.execute(
            "SELECT last_error FROM impact_errors"
        ).fetchone()[0]
    assert code == "EVIDENCE_MISMATCH"
    assert "secret" not in code
