"""Multi-year SEC source hashes and retrospective prices remain strictly separate."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest

from tris_market_event_intelligence.sec_multiyear_price_study_cli import (
    _approved_url,
    _validated_inputs,
    build_multiyear_sec_study,
)

REVIEW = Path("research/nvda-multiyear-sec-eight-k-official-review-2018-2026.json")


def _inputs(tmp_path: Path) -> tuple[dict, dict, Path, Path]:
    doc = json.loads(REVIEW.read_text())
    entries = [e for e in doc["entries"] if e["filingDate"] in (
        "2018-02-08", "2025-01-17",
    )]
    doc["entries"] = entries
    inventory = {
        "schemaVersion": "sec_historical_filing_inventory_v1",
        "companyCik": "0001045810",
        "ticker": "NVDA",
        "trainingReady": False,
        "candidateFilings": 2,
        "filings": [
            {"accession": e["accession"], "filingDate": e["filingDate"], "form": "8-K"}
            for e in entries
        ],
    }
    dates = [
        "2018-02-07", "2018-02-08", "2018-02-09", "2018-02-12",
        "2018-02-13", "2018-02-14", "2018-02-15",
        "2025-01-16", "2025-01-17", "2025-01-21",
        "2025-01-22", "2025-01-23", "2025-01-24",
        "2025-01-27",
    ]
    stock = tmp_path / "nvda.csv"
    benchmark = tmp_path / "spy.csv"
    stock.write_text(
        "date,adjusted_close,volume\n"
        + "".join(f"{day},{100 + i * 3},100\n" for i, day in enumerate(dates))
    )
    benchmark.write_text(
        "date,adjusted_close,volume\n"
        + "".join(f"{day},{100 + i},100\n" for i, day in enumerate(dates))
    )
    return doc, inventory, stock, benchmark


def _source_bytes(event: dict, *, index: bool) -> bytes:
    if not index:
        return b"<html><body>NVIDIA fiscal quarter revenue 2020 " + b"results" * 75 + b"</body></html>"
    category = event["category"]
    item = {"EARNINGS": "Item 2.02", "OTHER_EVENT": "Item 8.01",
            "GOVERNANCE": "Item 5.02",
            "CONTRACT_TERMINATION": "Item 1.02"}[category]
    accepted = event["secAcceptedEastern"][:19].replace("T", " ")
    exhibit_link = (
        '<a href="' + urlsplit(event["exhibit99_1Url"]).path.rsplit("/", 1)[-1]
        + '">EX-99.1</a>'
        if category == "EARNINGS" else ""
    )
    text = (
        f"<html><body>Form 8-K SEC Accession No. {event['accession']} "
        f"Filing Date {event['filingDate']} Accepted {accepted} "
        f"{item} {exhibit_link}</body></html>"
    )
    return text.encode()


def _client(review: dict, calls: list[str]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        event = next(e for e in review["entries"] if str(request.url) in (
            e["indexUrl"], e["exhibit99_1Url"],
        ))
        return httpx.Response(
            200,
            content=_source_bytes(event, index=str(request.url) == event["indexUrl"]),
        )
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_two_year_sources_pinned_and_prices_are_retrospective(tmp_path: Path) -> None:
    review, inventory, stock, benchmark = _inputs(tmp_path)
    requests: list[str] = []
    with _client(review, requests) as client:
        report = build_multiyear_sec_study(
            review=review, review_sha="a" * 64,
            inventory=inventory, inventory_sha="b" * 64,
            stock_csv=stock, benchmark_csv=benchmark, raw_dir=tmp_path,
            client=client, user_agent="Tris research contact@example.com",
            persist_raw=True,
        )
    assert report["eventCount"] == 2
    assert report["earningsEventCount"] == 1
    assert report["nonEarningsEventCount"] == 1
    assert report["rawSourceCount"] == 3
    assert len(requests) == 3
    assert report["trainingReady"] is False
    assert report["retrospectiveOutcomeAlreadyKnown"] is True
    for event in report["events"]:
        assert len(event["secIndexSha256"]) == 64
        assert event["trainingEligible"] is False
        assert event["outcome"]["includesMovementBeforeDecision"] is True
        assert set(event["outcome"]["horizonsTradingSessions"]) == {"1", "3", "5"}
        assert "stockReturn" not in json.dumps(event["features"])
        assert event["sourceAvailableByDecisionCutoff"] is False
        assert event["secReceiptBeforeDecisionCutoff"] is True
        assert event["publisherFirstReleaseTimeVerified"] is False
    assert report["events"][0]["category"] == "EARNINGS"
    assert report["events"][1]["category"] == "OTHER_EVENT"
    assert report["events"][1]["secExhibit99_1Sha256"] is None
    assert (tmp_path / "sec-0001045810-18-000004-index.html").exists()
    assert (tmp_path / "sec-0001045810-18-000004-exhibit99-1.html").exists()
    assert (tmp_path / "sec-0001045810-25-000007-index.html").exists()


def test_cached_sources_verify_offline_and_tamper_fails(tmp_path: Path) -> None:
    review, inventory, stock, benchmark = _inputs(tmp_path)
    with _client(review, []) as client:
        first = build_multiyear_sec_study(
            review=review, review_sha="a" * 64,
            inventory=inventory, inventory_sha="b" * 64,
            stock_csv=stock, benchmark_csv=benchmark, raw_dir=tmp_path,
            client=client, user_agent="Tris research contact@example.com",
            persist_raw=True,
        )
    second = build_multiyear_sec_study(
        review=review, review_sha="a" * 64,
        inventory=inventory, inventory_sha="b" * 64,
        stock_csv=stock, benchmark_csv=benchmark, raw_dir=tmp_path,
    )
    assert first["events"] == second["events"]
    index = tmp_path / "sec-0001045810-18-000004-index.html"
    index.write_bytes(index.read_bytes().replace(b"16:23:23", b"16:23:24"))
    with pytest.raises(ValueError, match="SEC_INDEX_FACTS_OR_ACCEPTANCE_MISMATCH"):
        build_multiyear_sec_study(
            review=review, review_sha="a" * 64,
            inventory=inventory, inventory_sha="b" * 64,
            stock_csv=stock, benchmark_csv=benchmark, raw_dir=tmp_path,
        )


def test_preview_without_raw_sources_fails_closed(tmp_path: Path) -> None:
    review, inventory, stock, benchmark = _inputs(tmp_path)
    with pytest.raises(ValueError, match="SEC_RAW_INPUT_MISSING"):
        build_multiyear_sec_study(
            review=review, review_sha="a" * 64,
            inventory=inventory, inventory_sha="b" * 64,
            stock_csv=stock, benchmark_csv=benchmark, raw_dir=tmp_path,
        )


def test_unverified_cross_issuer_or_future_clock_rejected(tmp_path: Path) -> None:
    review, inventory, _, _ = _inputs(tmp_path)
    assert len(_validated_inputs(review, inventory)) == 2
    review["entries"][0]["secAcceptedUtc"] = "2018-02-08T20:23:23+00:00"
    with pytest.raises(ValueError, match="SEC_EVENT_ACCEPTANCE_IN_INVALID_SESSION_WINDOW"):
        _validated_inputs(review, inventory)
    review, inventory, _, _ = _inputs(tmp_path)
    review["entries"][0]["indexUrl"] = "https://evil.example/Archives/malicious"
    with pytest.raises(ValueError, match="SEC_EVENT_DOCUMENT_CONTRACT_INVALID"):
        _validated_inputs(review, inventory)
    review, inventory, _, _ = _inputs(tmp_path)
    inventory["filings"][0]["form"] = "10-Q"
    with pytest.raises(ValueError, match="SEC_EVENT_INVENTORY_MISMATCH"):
        _validated_inputs(review, inventory)


def test_sec_url_allowlist_blocks_path_traversal_or_queries() -> None:
    acc = "0001045810-18-000004"
    base = "https://www.sec.gov/Archives/edgar/data/1045810/"
    assert _approved_url(
        base + "000104581018000004/" + acc + "-index.htm", acc, index=True,
    )
    for url in (
        base + "000104581018000004/../../evil.htm",
        base + "000104581018000004/%2e%2e%2fevil.htm",
        "https://evil.example/Archives/edgar/data/1045810/" + acc + "-index.htm",
        base + "000104581018000004/q4fy18pr.htm?secret=1",
    ):
        assert not _approved_url(url, acc, index=False)


def test_preopen_2019_and_2022_join_same_session(tmp_path: Path) -> None:
    review = json.loads(Path(
        "research/nvda-multiyear-sec-eight-k-official-review-2017-2026-v2.json"
    ).read_text())
    review["entries"] = [event for event in review["entries"]
                         if event["filingDate"] in ("2019-01-28", "2022-02-08")]
    inventory = {
        "schemaVersion": "sec_historical_filing_inventory_v1",
        "companyCik": "0001045810", "ticker": "NVDA",
        "trainingReady": False, "candidateFilings": 2,
        "filings": [{"accession": e["accession"], "filingDate": e["filingDate"],
                     "form": "8-K"} for e in review["entries"]],
    }
    dates = [
        "2019-01-24", "2019-01-25", "2019-01-28", "2019-01-29",
        "2019-01-30", "2019-01-31", "2019-02-01",
        "2022-02-04", "2022-02-07", "2022-02-08", "2022-02-09",
        "2022-02-10", "2022-02-11", "2022-02-14",
    ]
    stock, market = tmp_path / "nvda.csv", tmp_path / "spy.csv"
    for path, step in ((stock, 2), (market, 1)):
        path.write_text(
            "date,adjusted_close,volume\n" +
            "".join(f"{day},{100 + step * i},100\n"
                    for i, day in enumerate(dates))
        )
    assert len(_validated_inputs(review, inventory)) == 2
    calls: list[str] = []
    with _client(review, calls) as client:
        result = build_multiyear_sec_study(
            review=review, review_sha="a" * 64,
            inventory=inventory, inventory_sha="b" * 64,
            stock_csv=stock, benchmark_csv=market, raw_dir=tmp_path,
            client=client, user_agent="Tris research contact@example.com",
            persist_raw=True,
        )
    assert result["eventCount"] == 2
    assert result["earningsEventCount"] == 1
    assert result["nonEarningsEventCount"] == 1
    assert result["rawSourceCount"] == 3
    assert len(calls) == 3
    for event in result["events"]:
        assert event["eventSession"] == event["secAcceptedAt"][:10]
        assert event["secReceiptTiming"] == "PREOPEN_SAME_SESSION"
        assert event["sourceAvailableByDecisionCutoff"] is False
        assert event["independentPublicAvailabilityAtCutoffVerified"] is False
        assert event["decisionAt"] > event["secAcceptedAt"]
        assert event["trainingEligible"] is False
    assert result["events"][0]["features"]["previousSession"] == "2019-01-25"
    assert result["events"][1]["features"]["previousSession"] == "2022-02-07"
    assert result["events"][1]["category"] == "CONTRACT_TERMINATION"


def test_mid_session_sec_filing_cannot_be_backdated(tmp_path: Path) -> None:
    review, inventory, _, _ = _inputs(tmp_path)
    review["entries"][0]["secAcceptedEastern"] = "2018-02-08T13:30:00-05:00"
    review["entries"][0]["secAcceptedUtc"] = "2018-02-08T18:30:00+00:00"
    with pytest.raises(ValueError, match="SEC_EVENT_ACCEPTANCE_IN_INVALID_SESSION_WINDOW"):
        _validated_inputs(review, inventory)


def test_original_review_manifest_remains_unchanged() -> None:
    prior = json.loads(REVIEW.read_text())
    expanded = json.loads(Path(
        "research/nvda-multiyear-sec-eight-k-official-review-2017-2026-v2.json"
    ).read_text())
    assert len(prior["entries"]) == 7
    assert len(expanded["entries"]) == 10
    assert expanded["remainingYearFirstEightKUnreviewed"] == []
    for entry in prior["entries"]:
        assert entry in expanded["entries"]
    assert expanded["rawIndexAndExhibitSha256Pinned"] is False
    assert expanded["trainingReady"] is False
