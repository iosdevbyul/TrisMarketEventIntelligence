"""Batch prior-outcome historic references are resumable and never training Gold."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from test_historical_gdelt_scout import _archive, _study

from tris_market_event_intelligence.historical_gdelt_replay_cli import (
    replay_archive_coverage,
)


def test_preview_never_issues_network_requests(tmp_path: Path) -> None:
    study = _study(tmp_path)
    result = replay_archive_coverage(
        study=study, output_dir=tmp_path, start_rank=1, count=2,
    )
    assert result["status"] == "PREVIEW_ONLY"
    assert result["archiveSlotsEach"] == 4
    assert result["newReceipts"] == 0
    assert result["entries"][0]["action"] == "WOULD_FETCH_FOUR_SLOTS"
    assert result["trainingReady"] is False


def test_two_real_archive_rank_receipts_can_resume_without_re_fetch(
    tmp_path: Path,
) -> None:
    study = _study(tmp_path)
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, content=_archive())

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        first = replay_archive_coverage(
            study=study, output_dir=tmp_path, start_rank=1, count=2,
            run=True, client=client,
        )
        assert first["newReceipts"] == 2
        assert first["referenceCandidates"] == 8
        assert len(calls) == 8
        paths = [tmp_path / f"nvda-gdelt-gkg-rank{i}.json" for i in (1, 2)]
        originals = [p.read_bytes() for p in paths]
        second = replay_archive_coverage(
            study=study, output_dir=tmp_path, start_rank=1, count=2,
            run=True, client=client,
        )
    assert second["newReceipts"] == 0
    assert second["reusedReceipts"] == 2
    assert len(calls) == 8
    assert [p.read_bytes() for p in paths] == originals
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in paths)
    assert second["publisherTimesVerified"] is False
    assert second["trainingReady"] is False


def test_prior_receipt_must_match_exact_study_and_rank(tmp_path: Path) -> None:
    study = _study(tmp_path)
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=_archive()),
    )) as client:
        replay_archive_coverage(
            study=study, output_dir=tmp_path, start_rank=1, count=1,
            run=True, client=client,
        )
    receipt = tmp_path / "nvda-gdelt-gkg-rank1.json"
    data = json.loads(receipt.read_text())
    data["studySha256"] = "0" * 64
    receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="pinned study"):
        replay_archive_coverage(
            study=study, output_dir=tmp_path, start_rank=1, count=1,
        )


def test_rejects_fake_first_article_time_or_issuer_proof(tmp_path: Path) -> None:
    study = _study(tmp_path)
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=_archive()),
    )) as client:
        replay_archive_coverage(
            study=study, output_dir=tmp_path, start_rank=1, count=1,
            run=True, client=client,
        )
    receipt = tmp_path / "nvda-gdelt-gkg-rank1.json"
    data = json.loads(receipt.read_text())
    data["archives"][0]["matchingDocumentReferences"][0]["issuerVerified"] = True
    receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="causality"):
        replay_archive_coverage(
            study=study, output_dir=tmp_path, start_rank=1, count=1,
        )


def test_rank_and_directory_guardrails(tmp_path: Path) -> None:
    study = _study(tmp_path)
    with pytest.raises(ValueError, match="bounded"):
        replay_archive_coverage(
            study=study, output_dir=tmp_path, start_rank=1, count=100,
        )
    with pytest.raises(ValueError, match="bounded"):
        replay_archive_coverage(
            study=study, output_dir=tmp_path, start_rank=100, count=2,
        )
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="directory"):
        replay_archive_coverage(
            study=study, output_dir=link, start_rank=1, count=1,
        )
