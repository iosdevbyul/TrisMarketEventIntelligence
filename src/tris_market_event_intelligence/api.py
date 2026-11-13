"""HTTP interface. Event endpoints are introduced in later phases."""

from fastapi import FastAPI

from tris_market_event_intelligence import __version__


def create_app() -> FastAPI:
    app = FastAPI(title="TrisMarketEventIntelligence", version=__version__)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    return app


app = create_app()
