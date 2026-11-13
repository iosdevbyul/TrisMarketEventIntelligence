"""Runtime configuration with loopback-only defaults."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8100

    @classmethod
    def from_environment(cls) -> "Settings":
        host = os.getenv("MARKET_EVENT_HOST", "127.0.0.1")
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("MARKET_EVENT_HOST must be loopback-only")
        port = int(os.getenv("MARKET_EVENT_PORT", "8100"))
        if not 1 <= port <= 65535:
            raise ValueError("MARKET_EVENT_PORT must be between 1 and 65535")
        return cls(host=host, port=port)
