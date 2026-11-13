import pytest

from tris_market_event_intelligence.config import Settings


def test_default_configuration_is_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MARKET_EVENT_HOST", raising=False)
    monkeypatch.delenv("MARKET_EVENT_PORT", raising=False)
    assert Settings.from_environment() == Settings()


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.100", "example.com"])
def test_rejects_non_loopback_hosts(
    monkeypatch: pytest.MonkeyPatch, host: str
) -> None:
    monkeypatch.setenv("MARKET_EVENT_HOST", host)
    with pytest.raises(ValueError, match="loopback-only"):
        Settings.from_environment()


@pytest.mark.parametrize("port", ["0", "65536"])
def test_rejects_invalid_ports(monkeypatch: pytest.MonkeyPatch, port: str) -> None:
    monkeypatch.setenv("MARKET_EVENT_PORT", port)
    with pytest.raises(ValueError, match="between 1 and 65535"):
        Settings.from_environment()
