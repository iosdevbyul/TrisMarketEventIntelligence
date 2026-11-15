"""Historical GDELT research reads only bounded archive metadata, never labels."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import httpx
import pytest
from test_historical_shock_research import _setup

from tris_market_event_intelligence.historical_gdelt_scout import scout_historical_gdelt
from tris_market_event_intelligence.historical_shock_research import (
    build_historical_shock_research,
)


def _study(tmp_path: Path) -> Path:
    stock, benchmark, _, news = _setup(tmp_path)
    report = build_historical_shock_research(
        symbol="NVDA", benchmark_symbol="SPY", stock_path=stock,
        benchmark_path=benchmark, news_cohort_path=news,
        top=2, controls=2,
    )
    # Keep production research contract (exact 100 shock rows) for scout.
    # Unit-fixture has 44 candidate sessions; pad only fixture with unique
    # valid dates, never present these as real market data.
    while len(report["topEvents"]) < 100:
        row = dict(report["topEvents"][0])
        row["date"] = f"2026-{1 + len(report['topEvents']) // 31:02d}-{1 + len(report['topEvents']) % 28:02d}"
        report["topEvents"].append(row)
    report["topCount"] = 100
    path = tmp_path / "study.json"
    path.write_text(json.dumps(report))
    return path


def _archive() -> bytes:
    content = (
        b"20230524130000-1\t20230524130000\t1\tnews.example\t"
        b"https://news.example/nvda\t\t\t\t\t\t\t\t\tNvidia Corporation;\tNvidia,123;\n"
        b"20230524130000-2\t20230524130000\t1\tnews.example\t"
        b"https://news.example/other\t\t\t\t\t\t\t\t\tAnother company\t\n"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("20230524130000.gkg.csv", content)
    return buffer.getvalue()


def test_preview_uses_only_existing_pinned_packet_without_network(tmp_path: Path) -> None:
    study = _study(tmp_path)
    result = scout_historical_gdelt(study_path=study)
    assert result["status"] == "PREVIEW_ONLY"
    assert result["archivesFetched"] == 0
    assert len(result["archivePlan"]) == 4
    assert result["shockRank"] == 1
    assert result["sourceArticleRightsVerified"] is False
    assert result["predictionOrTrainingPerformed"] is False
    assert result["earliestArticleAvailabilityVerified"] is False


def test_bounded_zip_extracts_only_gkg_matching_reference(tmp_path: Path) -> None:
    study = _study(tmp_path)
    seen = []
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.url.host == "data.gdeltproject.org"
        assert request.url.path.endswith(".gkg.csv.zip")
        return httpx.Response(200, content=_archive())
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = scout_historical_gdelt(study_path=study, run=True, client=client)
    assert len(seen) == 4
    assert result["archivesFetched"] == 4
    for archive in result["archives"]:
        assert archive["matchingDocumentsTotal"] == 1
        reference = archive["matchingDocumentReferences"][0]
        assert reference["sourceUrl"] == "https://news.example/nvda"
        assert reference["articleTextRetrieved"] is False
        assert reference["issuerVerified"] is False
        assert archive["earliestArticleAvailabilityVerified"] is False
    assert result["economicImpactCausallyVerified"] is False
    assert result["predictionOrTrainingPerformed"] is False
    assert result["modelCalls"] == 0


def test_incorrect_packet_and_rank_are_rejected(tmp_path: Path) -> None:
    study = _study(tmp_path)
    with pytest.raises(ValueError, match="rank"):
        scout_historical_gdelt(study_path=study, rank=0)
    doc = json.loads(study.read_text())
    doc["forwardPredictionSealed"] = True
    study.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="retrospective"):
        scout_historical_gdelt(study_path=study)


def test_http_failure_contains_no_untrusted_raw_response(tmp_path: Path) -> None:
    study = _study(tmp_path)
    with (httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(403, text="PRIVATE AUTHORIZATION DETAILS")
    )) as client, pytest.raises(ValueError, match="GDELT_ARCHIVE_FETCH_FAILED") as err):
        scout_historical_gdelt(study_path=study, run=True, client=client)
    assert "PRIVATE AUTHORIZATION DETAILS" not in str(err.value)
