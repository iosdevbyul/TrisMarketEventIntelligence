"""Case 00 contextual antecedent and no-causal-guess regression tests."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
from test_sec_primary_semantic_review import _packet, _Provider

from tris_market_event_intelligence.sec_claim_evidence_review_cli import (
    build_claim_review,
)
from tris_market_event_intelligence.sec_pinned_body_spans import (
    CONTEXT_VERSION,
    VERSION,
    apply_body_spans,
)
from tris_market_event_intelligence.sec_primary_content_research_cli import (
    _source_review,
)
from tris_market_event_intelligence.sec_primary_semantic_research_cli import (
    prepare_primary_cases,
    run_semantic_research,
)
from tris_market_event_intelligence.sec_primary_semantic_review_cli import (
    build_review_packet,
)
from tris_market_event_intelligence.sec_revenue_context_guard import (
    DESCRIPTIVE_ONLY,
    REFERENT_UNRESOLVED,
    reject_unsupported_revenue_causes,
)
from tris_market_event_intelligence.sec_semantic_paired_comparison_cli import (
    compare_semantic_reports,
)


def _source(directory: Path, statement: str) -> dict:
    packet = _packet(directory, count=1)
    case = packet["cases"][0]
    # A genuine first-1100-character administrative cover followed by the
    # unmodified revenue-share fact. Never introduce new source claims.
    cover = "NVIDIA CORPORATION. " + (
        "FORM 10-K ANNUAL REPORT PURSUANT TO SECTION 13. " * 27
    )
    body = (
        "<html><body><p>" + cover + "</p>"
        "<p>Details of the issuer filing appear below.</p>"
        "<p>" + statement + "</p></body></html>"
    ).encode("utf-8")
    (directory / f"sec-{case['accession']}-primary.source").write_bytes(body)
    case["primaryDocumentSha256"] = hashlib.sha256(body).hexdigest()
    case.update(_source_review(body, case["primaryDocumentUrl"]))
    assert statement not in case["primarySourceVerbatimExcerpt"]
    return packet


def _reports(directory: Path, statement: str) -> tuple[dict, dict, dict]:
    primary = _source(directory, statement)
    original = prepare_primary_cases(primary, "a" * 64, directory)
    legacy = run_semantic_research(
        original, _Provider(original["cases"]), limit=1,
    )
    contextual = apply_body_spans(original, directory)
    provider = _Provider(contextual["cases"])
    current = run_semantic_research(
        contextual, provider, evidence_gated=True, limit=1,
    )
    assert provider.calls == []
    assert current["modelCalls"] == 0
    assert current["hypothesesCorrectionCount"] == 0
    assert current["casesAbstained"] == 1
    assert contextual["bodySpanSelectionVersion"] == CONTEXT_VERSION
    return primary, legacy, current


def test_case00_orphan_12_percent_abstains_and_preserves_context(
    tmp_path: Path,
) -> None:
    statement = "accounted for 12% of our total revenue for fiscal year 2017 ."
    primary, legacy, current = _reports(tmp_path, statement)
    row = current["cases"][0]
    assert row["analysisStatus"] == REFERENT_UNRESOLVED
    assert row["interpretation"]["hypotheses"] == []
    assert row["interpretation"]["observations"] == []
    span = row["sourceSpan"]
    assert span["contextStatus"] == REFERENT_UNRESOLVED
    assert span["focusStart"] >= 1100
    assert span["visibleStart"] < span["focusStart"]
    assert span["visibleEnd"] == span["focusEnd"]
    assert len(span["focusSha256"]) == 64
    review = build_review_packet(primary, "a" * 64, tmp_path, [(current, "c" * 64)])
    assert review["cases"][0]["sourceSpan"] == span
    assert statement in review["cases"][0]["sourceExcerpt"]
    assert review["cases"][0]["hypotheses"] == []
    assert review["modelCallsObserved"] == 0
    claims = build_claim_review(primary, "a" * 64, tmp_path, [(current, "c" * 64)])
    assert claims["claimCount"] == 0
    assert claims["abstainedCaseIndexes"] == [0]
    paired = compare_semantic_reports(
        primary, "a" * 64, tmp_path,
        [(legacy, "b" * 64)], [(current, "c" * 64)],
    )
    assert paired["pairs"][0]["evidenceScopeChanged"] is True
    assert paired["semanticQualityImprovementVerified"] is False


def test_direct_customer_referent_remains_descriptive_not_causal(
    tmp_path: Path,
) -> None:
    primary, _, current = _reports(
        tmp_path,
        "One customer accounted for 12% of our total revenue for fiscal year 2017.",
    )
    assert current["cases"][0]["analysisStatus"] == DESCRIPTIVE_ONLY
    assert current["cases"][0]["sourceSpan"]["contextStatus"] == DESCRIPTIVE_ONLY
    assert current["cases"][0]["interpretation"]["hypotheses"] == []
    assert "a directly stated referent" in current["cases"][0]["interpretation"]["summary"]
    build_review_packet(primary, "a" * 64, tmp_path, [(current, "c" * 64)])


@pytest.mark.parametrize("assertion", [
    "Management made a deliberate strategic decision to prioritize high-margin sales.",
    "The 12% figure arose from internal resource reallocation and strategic focus.",
    "This was due to macroeconomic shifts and sector-wide trends during 2017.",
    "The company decided on a targeted growth strategy for this business.",
])
def test_hypothetical_strategy_cannot_be_inferred_from_revenue_ratio(
    assertion: str,
) -> None:
    interpretation = {
        "summary": "A historical percentage appears in the source.",
        "observations": [{
            "interpretation": "The filing reports a historical revenue proportion.",
        }],
        "hypotheses": [
            {"category": "ISSUER_FUNDAMENTALS",
             "mechanism": assertion,
             "limitation": "No internal records have been supplied."},
        ],
    }
    with pytest.raises(ValueError, match="SEC_REVENUE_UNSUPPORTED_CAUSAL_NARRATIVE"):
        reject_unsupported_revenue_causes(
            "One customer accounted for 12% of our total revenue for fiscal year 2017.",
            interpretation,
        )


@pytest.mark.parametrize("changed", ["visibleStart", "focusStart", "focusEnd"])
def test_changed_context_offsets_fail_closed(
    tmp_path: Path, changed: str,
) -> None:
    primary, _, current = _reports(
        tmp_path, "accounted for 12% of our total revenue for fiscal year 2017 .",
    )
    forged = copy.deepcopy(current)
    forged["cases"][0]["sourceSpan"][changed] += 1
    with pytest.raises(ValueError, match="SEC_BODY_SPAN_PROVENANCE_INVALID"):
        build_review_packet(primary, "a" * 64, tmp_path, [(forged, "c" * 64)])


def test_forged_analysis_status_does_not_override_sha_context(
    tmp_path: Path,
) -> None:
    primary, _, current = _reports(
        tmp_path, "accounted for 12% of our total revenue for fiscal year 2017 .",
    )
    forged = copy.deepcopy(current)
    forged["cases"][0]["analysisStatus"] = DESCRIPTIVE_ONLY
    with pytest.raises(ValueError, match="SEC_SEMANTIC_ABSTENTION_PROVENANCE_INVALID"):
        build_review_packet(primary, "a" * 64, tmp_path, [(forged, "c" * 64)])


def test_v1_filing_body_report_still_revalidates_without_context_status(
    tmp_path: Path,
) -> None:
    original = _source(
        tmp_path,
        "NVIDIA reported revenue of $5.2 billion during this fiscal period.",
    )
    prepared = prepare_primary_cases(original, "a" * 64, tmp_path)
    v1 = apply_body_spans(prepared, tmp_path, version=VERSION)
    report = run_semantic_research(
        v1, _Provider(v1["cases"]), evidence_gated=True, limit=1,
    )
    assert report["bodySpanSelectionVersion"] == VERSION
    assert report["casesAbstained"] == 0
    assert "contextStatus" not in report["cases"][0]["sourceSpan"]
    review = build_review_packet(original, "a" * 64, tmp_path, [(report, "b" * 64)])
    assert review["cases"][0]["sourceSpan"]["version"] == VERSION
