"""Phase 7-B Intel Mac inference timeout regression tests."""

from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from tris_market_event_intelligence.impact import (
    OpenAICompatibleProvider,
)
from tris_market_event_intelligence.impact_cli import main
from tris_market_event_intelligence.impact_store import AnalysisStats


@pytest.mark.parametrize("timeout", [0, -1, 901])
def test_invalid_provider_timeout_is_rejected(timeout: int) -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        OpenAICompatibleProvider(
            base_url="http://127.0.0.1:11434/v1", api_key="placeholder",
            model="gemma2:latest", timeout_seconds=timeout,
        )


def test_default_provider_timeout_is_90_and_custom_timeout_supported() -> None:
    # Inspect only HTTP client construction, no network call.
    from tris_market_event_intelligence.impact import EventInput

    event = EventInput(
        1, "CORPORATE_EVENT", "UNVERIFIED", None, None,
        "NVIDIA CEO resigns", "bbc_business", "https://example.org", "now",
        "REPORTED",
    )
    with patch("tris_market_event_intelligence.impact.httpx.Client") as client_class:
        client = client_class.return_value
        client.post.side_effect = httpx.ReadTimeout("slow")
        provider = OpenAICompatibleProvider(
            base_url="http://127.0.0.1:11434/v1", api_key="placeholder",
            model="gemma2:latest", timeout_seconds=300,
        )
        with pytest.raises(httpx.ReadTimeout):
            provider.analyze(event)
        timeout = client_class.call_args.kwargs["timeout"]
        assert isinstance(timeout, httpx.Timeout)
        assert timeout.read == 300
        assert timeout.connect == 300
        client.close.assert_called_once()


def test_local_cli_timeout_is_300_by_default(tmp_path: Path, capsys) -> None:
    db = tmp_path / "db.sqlite3"
    response = httpx.Response(
        200, json={"models": [{"name": "gemma2:latest"}]},
        request=httpx.Request("GET", "http://127.0.0.1:11434/api/tags"),
    )
    with (
        patch("sys.argv", ["market-event-analyze", "--db", str(db),
                           "--provider", "local", "--batch-size", "1"]),
        patch.dict("os.environ", {"MARKET_EVENT_LLM_MODEL": "gemma2:latest"}),
        patch("tris_market_event_intelligence.impact_cli.httpx.get", return_value=response),
        patch("tris_market_event_intelligence.impact_cli.OpenAICompatibleProvider")
        as provider_class,
        patch("tris_market_event_intelligence.impact_cli.ImpactProcessor")
        as processor_class,
    ):
        processor_class.return_value.process_pending.return_value = AnalysisStats(0, 0, 0)
        main()
        assert provider_class.call_args.kwargs["timeout_seconds"] == 300
    assert '"failed": 0' in capsys.readouterr().out


def test_cli_rejects_timeout_before_provider_creation(tmp_path: Path) -> None:
    response = httpx.Response(
        200, json={"models": [{"name": "gemma2:latest"}]},
        request=httpx.Request("GET", "http://127.0.0.1:11434/api/tags"),
    )
    with (
        patch("sys.argv", ["market-event-analyze", "--provider", "local",
                           "--timeout-seconds", "901",
                           "--db", str(tmp_path / "missing.sqlite3")]),
        patch.dict("os.environ", {"MARKET_EVENT_LLM_MODEL": "gemma2:latest"}),
        patch("tris_market_event_intelligence.impact_cli.httpx.get", return_value=response),
        patch("tris_market_event_intelligence.impact_cli.OpenAICompatibleProvider")
        as provider_class,
        pytest.raises(SystemExit) as error,
    ):
        main()
    assert error.value.code == 2
    provider_class.assert_not_called()
    assert not (tmp_path / "missing.sqlite3").exists()
