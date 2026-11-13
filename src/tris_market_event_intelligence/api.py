"""Loopback-only, read-only consumer API for independent stock analysis systems."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

from tris_market_event_intelligence import __version__
from tris_market_event_intelligence.api_queries import (
    EventReader,
    claims_database_ready,
    database_ready,
)


def create_app(db_path: Path | None = None) -> FastAPI:
    app = FastAPI(title="TrisMarketEventIntelligence", version=__version__)
    path = db_path if db_path is not None else Path(
        os.getenv("MARKET_EVENT_DB", "data/news.sqlite3")
    )
    reader = EventReader(path)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/ready")
    def ready() -> dict[str, str]:
        if not database_ready(path):
            raise HTTPException(status_code=503, detail="Event database is not ready")
        return {"status": "ready"}

    @app.get("/ready/claims")
    def claims_ready() -> dict[str, str]:
        if not claims_database_ready(path):
            raise HTTPException(status_code=503, detail="Claim database is not ready")
        return {"status": "ready"}

    @app.get("/api/v1/events")
    def list_events(
        after_id: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=500),
        status: str | None = Query(default=None, pattern="^(VERIFIED|CANDIDATE|UNVERIFIED)$"),
    ) -> dict[str, object]:
        try:
            return reader.events(after_id=after_id, limit=limit, status=status)
        except (OSError, sqlite3.Error) as exc:
            raise HTTPException(status_code=503, detail="Event database unavailable") from exc

    @app.get("/api/v1/event-claims")
    def list_event_claims(
        after_id: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, object]:
        try:
            return reader.claims(after_id=after_id, limit=limit)
        except (OSError, sqlite3.Error) as exc:
            raise HTTPException(status_code=503, detail="Claim database unavailable") from exc

    @app.get("/api/v1/claim-assessments")
    def list_claim_assessments(
        after_id: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, object]:
        try:
            return reader.claim_assessments(after_id=after_id, limit=limit)
        except (OSError, sqlite3.Error) as exc:
            raise HTTPException(
                status_code=503, detail="Claim assessment database unavailable"
            ) from exc

    @app.get("/api/v1/claim-reviews")
    def list_claim_reviews(
        after_id: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, object]:
        try:
            return reader.latest_claim_reviews(after_id=after_id, limit=limit)
        except (OSError, sqlite3.Error) as exc:
            raise HTTPException(status_code=503, detail="Review database unavailable") from exc

    @app.get("/api/v1/claim-review-history")
    def list_claim_review_history(
        after_id: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=500),
        claim_id: int | None = Query(default=None, ge=1),
    ) -> dict[str, object]:
        try:
            return reader.claim_review_history(
                after_id=after_id, limit=limit, claim_id=claim_id
            )
        except (OSError, sqlite3.Error) as exc:
            raise HTTPException(status_code=503, detail="Review database unavailable") from exc

    @app.get("/api/v1/impacts")
    def list_impacts(
        after_id: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, object]:
        try:
            return reader.analyses(after_id=after_id, limit=limit)
        except (OSError, sqlite3.Error, ValueError) as exc:
            raise HTTPException(status_code=503, detail="Impact database unavailable") from exc

    @app.get("/api/v1/analysis-runs")
    def list_analysis_runs(
        after_id: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, object]:
        try:
            return reader.analysis_runs(after_rowid=after_id, limit=limit)
        except (OSError, sqlite3.Error) as exc:
            raise HTTPException(status_code=503, detail="Analysis database unavailable") from exc

    return app


app = create_app()
