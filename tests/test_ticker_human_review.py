"""Human ticker review is explicit, hash-pinned, and never training-ready."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from tris_market_event_intelligence.ticker_human_review import review_ticker_candidates
from tris_market_event_intelligence.ticker_human_review_cli import main

DETECTED = "2026-11-14T13:00:00+00:00"
REVIEWED = "2026-11-14T14:00:00+00:00"
CLOCK = datetime(2026, 11, 14, 15, 0, tzinfo=UTC)
FIELDS = (
    "event_id", "market", "symbol", "decision", "reviewed_at",
    "reviewer_id", "evidence_url", "review_note", "proposal_sha256",
)


def fixture(tmp_path: Path) -> tuple[Path, Path]:
    proposals = {
        "schema_version": "event_ticker_proposals_v1",
        "status": "REVIEW_REQUIRED",
        "registry_source_authenticated": False,
        "symbol_identity_independently_verified": False,
        "economic_relationship_verified": False,
        "point_in_time_registry_claims_verified": False,
        "training_dataset_eligible": False,
        "market_impact_predicted": False,
        "database_modified": False,
        "items": [{
            "event_id": 7, "status": "PROPOSED",
            "proposals": [{
                "market": "XNAS", "symbol": "MSFT",
                "relationship": "DIRECT_CANDIDATE",
                "human_review_required": True,
                "article_detected_at": DETECTED,
                "registry_evidence_seen_at": "2026-01-01T00:00:00+00:00",
                "article_url": "https://example.org/news/7",
                "source_id": "reviewed-list",
            }],
        }],
    }
    proposal_file = tmp_path / "proposals.json"
    proposal_file.write_text(json.dumps(proposals), encoding="utf-8")
    return proposal_file, tmp_path / "decisions.csv"


def write_decision(path: Path, proposal_file: Path, **overrides: str) -> None:
    decision = {
        "event_id": "7", "market": "XNAS", "symbol": "MSFT",
        "decision": "APPROVE", "reviewed_at": REVIEWED,
        "reviewer_id": "researcher-a",
        "evidence_url": "https://example.org/issuer-identity",
        "review_note": "Company legal name checked, research only",
        "proposal_sha256": hashlib.sha256(proposal_file.read_bytes()).hexdigest(),
        **overrides,
    }
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(decision)


def test_approval_remains_posthoc_and_never_generates_training_link(tmp_path: Path) -> None:
    proposals, decisions = fixture(tmp_path)
    write_decision(decisions, proposals)
    before = (proposals.read_bytes(), decisions.read_bytes())
    report = review_ticker_candidates(proposals, decisions, as_of=CLOCK)
    assert report["status"] == "REVIEWED_NOT_TRAINING_READY"
    assert report["approved_for_post_hoc_review"] == 1
    assert report["training_dataset_eligible"] is False
    assert report["auto_approved"] is False
    assert report["actual_reviewer_identity_authenticated"] is False
    assert report["external_issuer_evidence_authenticated"] is False
    item = report["items"][0]
    assert item["review_status"] == "POST_HOC_RESEARCH_ASSOCIATION"
    assert item["eligible_for_point_in_time_training"] is False
    assert "benchmark_symbol" not in item
    assert "evidence_seen_at" not in item
    assert before == (proposals.read_bytes(), decisions.read_bytes())


def test_rejected_candidate_is_not_mapped(tmp_path: Path) -> None:
    proposals, decisions = fixture(tmp_path)
    write_decision(decisions, proposals, decision="REJECT")
    report = review_ticker_candidates(proposals, decisions, as_of=CLOCK)
    assert report["rejected"] == 1
    assert report["approved_for_post_hoc_review"] == 0
    assert report["items"][0]["review_status"] == "REJECTED"


@pytest.mark.parametrize("changes", [
    {"symbol": "AAPL"},
    {"event_id": "8"},
    {"proposal_sha256": "a" * 64},
    {"decision": "AUTO"},
    {"reviewed_at": "2026-11-14T12:00:00+00:00"},
    {"reviewed_at": "2026-11-14T16:00:00+00:00"},
    {"reviewed_at": "2026-11-14T14:00:00"},
    {"evidence_url": "http://example.org/issuer"},
    {"evidence_url": "https://localhost/private"},
    {"reviewer_id": ""},
    {"review_note": ""},
])
def test_bad_decisions_fail_closed(tmp_path: Path, changes: dict[str, str]) -> None:
    proposals, decisions = fixture(tmp_path)
    write_decision(decisions, proposals, **changes)
    with pytest.raises(ValueError, match="row 1"):
        review_ticker_candidates(proposals, decisions, as_of=CLOCK)


def test_modified_proposals_fail_hash_match(tmp_path: Path) -> None:
    proposals, decisions = fixture(tmp_path)
    write_decision(decisions, proposals)
    document = json.loads(proposals.read_text())
    document["items"][0]["proposals"][0]["symbol"] = "AAPL"
    proposals.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="row 1"):
        review_ticker_candidates(proposals, decisions, as_of=CLOCK)


def test_future_listing_evidence_or_unverified_flags_fail(tmp_path: Path) -> None:
    proposals, decisions = fixture(tmp_path)
    write_decision(decisions, proposals)
    document = json.loads(proposals.read_text())
    document["items"][0]["proposals"][0][
        "registry_evidence_seen_at"
    ] = "2026-12-01T00:00:00+00:00"
    proposals.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="future registry evidence"):
        review_ticker_candidates(proposals, decisions, as_of=CLOCK)
    document["items"][0]["proposals"][0][
        "registry_evidence_seen_at"
    ] = "2026-01-01T00:00:00+00:00"
    document["training_dataset_eligible"] = True
    proposals.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="unverified flags"):
        review_ticker_candidates(proposals, decisions, as_of=CLOCK)


def test_symlinked_inputs_and_duplicate_decisions_fail(tmp_path: Path) -> None:
    proposals, decisions = fixture(tmp_path)
    write_decision(decisions, proposals)
    link = tmp_path / "linked.json"
    link.symlink_to(proposals)
    with pytest.raises(ValueError, match="non-symlink"):
        review_ticker_candidates(link, decisions, as_of=CLOCK)
    with decisions.open("a", encoding="utf-8") as stream:
        stream.write(decisions.read_text().splitlines()[-1] + "\n")
    with pytest.raises(ValueError, match="row 2"):
        review_ticker_candidates(proposals, decisions, as_of=CLOCK)


def test_cli_is_read_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    proposals, decisions = fixture(tmp_path)
    write_decision(decisions, proposals)
    before = set(tmp_path.iterdir())
    with patch.object(sys, "argv", [
        "market-event-ticker-review", "--proposals", str(proposals),
        "--decisions", str(decisions),
    ]):
        main()
    output = json.loads(capsys.readouterr().out)
    assert output["approved_for_post_hoc_review"] == 1
    assert output["existing_database_modified"] is False
    assert set(tmp_path.iterdir()) == before
