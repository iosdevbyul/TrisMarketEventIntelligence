"""Collected-source SQLite adapter for generic market pressure."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tris_market_event_intelligence.market_evidence_db_bridge import (
    make_packet_from_db,
)
from tris_market_event_intelligence.market_evidence_pressure import analyze_market_evidence


def populated_db(path: Path) -> Path:
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE articles (
                id INTEGER PRIMARY KEY, source TEXT, url TEXT, title TEXT,
                summary TEXT, published_at TEXT
            );
            CREATE TABLE economic_events (
                id INTEGER PRIMARY KEY, classification TEXT, evidence TEXT,
                source TEXT, source_url TEXT, published_at TEXT, status TEXT
            );
        """)
        conn.execute(
            "INSERT INTO articles VALUES (1, ?, ?, ?, ?, ?)",
            (
                "example-feed", "https://www.example.org/industry",
                "Industry costs may rise",
                "A published industry report described increased input costs.",
                "2024-02-14T10:00:00+00:00",
            ),
        )
        conn.execute(
            "INSERT INTO articles VALUES (2, ?, ?, ?, ?, ?)",
            (
                "future-feed", "https://www.example.org/future",
                "Later earnings release",
                "A future report mentioned changes not yet observable.",
                "2024-02-16T10:00:00+00:00",
            ),
        )
        conn.execute(
            "INSERT INTO economic_events VALUES (7, ?, ?, ?, ?, ?, ?)",
            (
                "ECONOMIC_INDICATOR",
                "An official indicator release documented a measured increase.",
                "indicator-feed", "https://www.bls.gov/news.release/cpi.toc.htm",
                "2024-02-13T14:00:00+00:00", "CANDIDATE",
            ),
        )
    return path


class EmptyProvider:
    model_id = "mock"

    def __init__(self) -> None:
        self.payloads: list[str] = []

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        self.payloads.append(user)
        return '{"impacts":[]}'


def test_db_adapter_reuses_collected_articles_and_indicators(tmp_path: Path) -> None:
    db = populated_db(tmp_path / "news.sqlite3")
    packet = make_packet_from_db(
        db, event_id="demo", subject="Costs and inflation",
        as_of="2024-02-15", article_ids=(1, 2),
        economic_event_ids=(7,),
    )
    assert [x.kind for x in packet.evidences] == [
        "NEWS", "NEWS", "ECONOMIC_INDICATOR",
    ]
    assert all(x.verification == "REPORTED_UNVERIFIED"
               for x in packet.evidences)
    p = EmptyProvider()
    result = analyze_market_evidence(packet, p)
    assert result["visibleSourceIds"] == ["article-1", "economic-event-7"]
    assert result["excludedFutureSourceIds"] == ["article-2"]
    assert result["modelCalls"] == 1
    assert len(p.payloads) == 1
    assert "Later earnings release" not in p.payloads[0]


def test_db_adapter_requires_dated_and_existing_original_sources(
    tmp_path: Path,
) -> None:
    db = populated_db(tmp_path / "news.sqlite3")
    with pytest.raises(ValueError, match="MARKET_SOURCE_ARTICLE_NOT_FOUND"):
        make_packet_from_db(
            db, event_id="demo", subject="Test", as_of="2024-02-15",
            article_ids=(999,),
        )
    with pytest.raises(ValueError, match="MARKET_SOURCE_SELECTION_INVALID"):
        make_packet_from_db(
            db, event_id="demo", subject="Test", as_of="2024-02-15",
            article_ids=(1, 1),
        )
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE articles SET published_at=NULL WHERE id=1")
    with pytest.raises(ValueError, match="PUBLICATION_DATE_MISSING"):
        make_packet_from_db(
            db, event_id="demo", subject="Test", as_of="2024-02-15",
            article_ids=(1,),
        )
