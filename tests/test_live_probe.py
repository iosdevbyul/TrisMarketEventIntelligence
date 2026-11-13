"""Mock transport validates the same live probe orchestration used by CI."""
import httpx

from tris_market_event_intelligence.live_probe import run_probe


def test_probe_records_real_collection_metrics_without_claiming_mac_success() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.gdeltproject.org":
            return httpx.Response(200, json={"articles": [{
                "title": "GPU supply shortage", "url": "https://example.org/gpu",
                "seendate": "20261114T000000Z",
            }]})
        return httpx.Response(200, content=b"""<rss><channel><item>
        <title>Consumer Price Index CPI released</title>
        <link>https://example.org/cpi</link><guid>1</guid></item></channel></rss>""")

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        result = run_probe(client)
    assert result["rss_success"] == 3
    assert result["rss_total"] == 3
    assert result["articles"] == 4
    assert result["claims"]["claims"] >= 4
    assert result["benchmark_observed"] == 0
    assert not result["gate_6_ready"]
    assert result["mode"] == "LIVE_GITHUB_RUNNER_NOT_LOCAL_MAC"


def test_probe_reports_network_failure_not_fake_success() -> None:
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(503)
    )) as client:
        result = run_probe(client)
    assert result["rss_success"] == 0
    assert result["articles"] == 0
    assert result["claims"]["claims"] == 0
    assert all(value["status"] == "error" for value in result["rss"].values())
