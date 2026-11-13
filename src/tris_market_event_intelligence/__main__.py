"""Start the local HTTP service."""

import uvicorn

from tris_market_event_intelligence.config import Settings


def main() -> None:
    settings = Settings.from_environment()
    uvicorn.run(
        "tris_market_event_intelligence.api:app",
        host=settings.host,
        port=settings.port,
    )


if __name__ == "__main__":
    main()
