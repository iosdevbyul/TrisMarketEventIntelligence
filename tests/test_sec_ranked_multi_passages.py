"""Multi-passage full original retrieval, evidence ranking and model grounding."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from test_sec_primary_semantic_review import _packet, _Provider

from tris_market_event_intelligence.sec_claim_evidence_review_cli import build_claim_review
from tris_market_event_intelligence.sec_multi_grounded_analysis import (
    generate_ranked_interpretation,
    validate_ranked_interpretation,
)
from tris_market_event_intelligence.sec_primary_content_research_cli import _source_review
from tris_market_event_intelligence.sec_primary_semantic_research_cli import (
    prepare_primary_cases,
    run_semantic_research,
)
from tris_market_event_intelligence.sec_primary_semantic_review_cli import build_review_packet
from tris_market_event_intelligence.sec_ranked_multi_passages import (
    VERSION,
    apply_ranked_passages,
)
from tris_market_event_intelligence.sec_semantic_candidate_replay_cli import (
    prepare_candidate_replay,
    replay_candidate,
)
from tris_market_event_intelligence.sec_semantic_paired_comparison_cli import (
    compare_semantic_reports,
)


class BoundedProvider:
    model_id = "offline-fixture"

    def __init__(self, *, invent: str | None = None):
        self.calls = 0
        self.invent = invent

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        self.calls += 1
        request = json.loads(user)
        evidences = request["sourceEvidence"]
        assert all("contextWindow" in evidence for evidence in evidences)
        assert all("quote_id" in evidence for evidence in evidences)
        assert len(evidences) >= 2
        assert schema["properties"]["hypotheses"]["maxItems"] == 0
        return json.dumps({
            "observations": [
                {
                    "quote_id": item["quote_id"],
                    "interpretation": (
                        self.invent or
                        "The filing contains this original issuer statement: "
                        + item["verbatimSourceText"]
                    ),
                }
                for item in evidences[:2]
            ],
            "hypotheses": [],
            "summary": "Two separate original issuer facts appear, without proving causality.",
        })


def _primary(tmp_path: Path, *, one: bool = False, orphan: bool = True) -> dict:
    packet = _packet(tmp_path, count=1)
    case = packet["cases"][0]
    cover = "NVIDIA CORPORATION. " + (
        "FORM 10-K ANNUAL REPORT PURSUANT TO SECTION 13. " * 28
    )
    fragment = (
        "accounted for 12% of our total revenue for fiscal year 2017."
        if orphan else
        "One customer accounted for 12% of our total revenue for fiscal year 2017."
    )
    filler = "Contextual information not sufficient to prove company causality. " * 4
    facts = [
        "NVIDIA reported revenue of $5.2 billion during fiscal 2017.",
    ]
    if not one:
        facts.extend([
            "NVIDIA launched a new computing platform to support its customers.",
            "NVIDIA relies on a limited number of suppliers for manufacturing components.",
        ])
    body = "<html><body>" + cover + " " + fragment
    for fact in facts:
        body += "<p>" + filler + fact + "</p>"
    body += "</body></html>"
    raw = body.encode("utf-8")
    path = tmp_path / f"sec-{case['accession']}-primary.source"
    path.write_bytes(raw)
    case["primaryDocumentSha256"] = hashlib.sha256(raw).hexdigest()
    case.update(_source_review(raw, case["primaryDocumentUrl"]))
    assert fragment not in case["primarySourceVerbatimExcerpt"]
    return packet


def _run(tmp_path: Path) -> tuple[dict, dict, dict, BoundedProvider]:
    packet = _primary(tmp_path)
    prepared = prepare_primary_cases(packet, "a" * 64, tmp_path)
    selected = apply_ranked_passages(prepared, tmp_path)
    model = BoundedProvider()
    report = run_semantic_research(selected, model, evidence_gated=True, limit=1)
    return packet, selected, report, model


def test_ranked_passages_skip_orphan_share_and_use_distinct_facts(tmp_path: Path) -> None:
    packet, selected, report, provider = _run(tmp_path)
    assert selected["multiPassageVersion"] == VERSION
    assert selected["multiPassageCasesSelected"] == 1
    case = selected["cases"][0]
    assert 2 <= len(case["sourcePassages"]) <= 3
    assert all(p["focusStart"] > 1100 for p in case["sourcePassages"])
    assert len({p["focusStart"] for p in case["sourcePassages"]}) >= 2
    assert all(p["originalSourceSha256"] == packet["cases"][0]["primaryDocumentSha256"]
               for p in case["sourcePassages"])
    assert all("accounted for 12%" not in e["focusText"] for e in case["evidence"])
    assert all(e["focusText"] in e["text"] for e in case["evidence"])
    assert provider.calls == 1
    assert report["modelCalls"] == 1
    assert report["casesAbstained"] == 0
    assert report["cases"][0]["interpretation"]["hypotheses"] == []
    assert len(report["cases"][0]["interpretation"]["observations"]) == 2
    assert report["trainingReady"] is False
    assert report["promotionAllowed"] is False


def test_multi_passage_review_claims_and_paired_scope(tmp_path: Path) -> None:
    packet, selected, report, _ = _run(tmp_path)
    reviewed = build_review_packet(packet, "a" * 64, tmp_path, [(report, "c" * 64)])
    entry = reviewed["cases"][0]
    assert entry["sourcePassages"] == selected["cases"][0]["sourcePassages"]
    assert len(entry["sourceExcerpts"]) >= 2
    for obs in entry["observations"]:
        matching = next(
            text for text in entry["sourceExcerpts"]
            if text["evidenceId"] == obs["evidence_id"]
        )
        assert obs["verbatim_quote"] in matching["text"]
    claims = build_claim_review(packet, "a" * 64, tmp_path, [(report, "c" * 64)])
    assert claims["claimCount"] == 3
    assert all(len(row["sourceExcerpts"]) >= 2 for row in claims["claims"])
    assert len(claims["claims"][0]["candidateEvidence"]) == 1
    assert len(claims["claims"][1]["candidateEvidence"]) == 1
    legacy_prepared = prepare_primary_cases(packet, "a" * 64, tmp_path)
    baseline = run_semantic_research(
        legacy_prepared, _Provider(legacy_prepared["cases"]), limit=1,
    )
    comparison = compare_semantic_reports(
        packet, "a" * 64, tmp_path,
        [(baseline, "b" * 64)], [(report, "c" * 64)],
    )
    pair = comparison["pairs"][0]
    assert pair["evidenceScopeChanged"] is True
    assert len(pair["candidatePassageSha256"]) >= 2
    assert comparison["semanticQualityImprovementVerified"] is False


def test_source_witness_mutation_rejected_and_old_primary_unchanged(tmp_path: Path) -> None:
    packet, _, report, _ = _run(tmp_path)
    mutated = copy.deepcopy(report)
    mutated["cases"][0]["sourcePassages"][0]["focusStart"] += 1
    with pytest.raises(ValueError, match="SEC_MULTI_PASSAGES_PROVENANCE_INVALID"):
        build_review_packet(packet, "a" * 64, tmp_path, [(mutated, "c" * 64)])
    omitted = copy.deepcopy(report)
    omitted["cases"][0]["sourcePassages"].pop()
    with pytest.raises(ValueError, match="SEC_MULTI_PASSAGES_PROVENANCE_INVALID"):
        build_review_packet(packet, "a" * 64, tmp_path, [(omitted, "c" * 64)])
    raw = tmp_path / f"sec-{packet['cases'][0]['accession']}-primary.source"
    raw.write_bytes(raw.read_bytes().replace(b"launched", b"altered_"))
    with pytest.raises(ValueError, match="SEC_SEMANTIC_SOURCE_SHA_MISMATCH"):
        build_review_packet(packet, "a" * 64, tmp_path, [(report, "c" * 64)])


def test_preparation_preserves_legacy_baseline_and_ranks_full_source(tmp_path: Path) -> None:
    packet = _primary(tmp_path)
    old = prepare_primary_cases(packet, "a" * 64, tmp_path)
    baseline = run_semantic_research(old, _Provider(old["cases"]), limit=1)
    selected, plan = prepare_candidate_replay(
        packet, "a" * 64, tmp_path, [(baseline, "b" * 64)],
        start_index=0, limit=1, multi_passages=True,
    )
    assert plan["multiPassagesEnabled"] is True
    assert plan["selectedPassagesCount"] >= 2
    assert plan["requestedBodySpansSelected"] == 0
    result = replay_candidate(selected, plan, BoundedProvider())
    assert result["cases"][0]["sourcePassages"]
    assert "sourcePassages" not in baseline["cases"][0]


def test_one_passage_safely_abstains_without_model(tmp_path: Path) -> None:
    packet = _primary(tmp_path, one=True)
    prepared = apply_ranked_passages(
        prepare_primary_cases(packet, "a" * 64, tmp_path), tmp_path,
    )
    assert prepared["multiPassageCasesSelected"] == 0
    provider = BoundedProvider()
    report = run_semantic_research(prepared, provider, evidence_gated=True, limit=1)
    assert provider.calls == 0
    assert report["casesAbstained"] == 1
    assert report["modelCalls"] == 0
    assert report["cases"][0]["interpretation"]["hypotheses"] == []
    assert report["cases"][0]["analysisStatus"] == "INSUFFICIENT_SUBSTANTIVE_EVIDENCE"
    build_review_packet(packet, "a" * 64, tmp_path, [(report, "c" * 64)])


def test_invented_numeric_fact_or_sector_driver_rejected(tmp_path: Path) -> None:
    _, selected, _, _ = _run(tmp_path)
    case = selected["cases"][0]
    with pytest.raises(ValueError, match="SEC_MULTI_UNSUPPORTED_NUMERIC_FACT"):
        generate_ranked_interpretation(case, BoundedProvider(invent="Sales grew 99% in 2026."))
    base = {
        "observations": [
            {
                "evidence_id": e["id"], "verbatim_quote": e["focusText"],
                "interpretation": "The issuer mentions this event only."
            }
            for e in case["evidence"][:2]
        ],
        "hypotheses": [
            {
                "category": "SECTOR_ALTERNATIVE",
                "evidence_ids": [case["evidence"][0]["id"]],
                "mechanism": "Sector-wide market contraction caused the revenue figure.",
                "limitation": "The relevant market evidence is not provided."
            }
        ],
        "summary": "Two issuer statements do not constitute independent sector facts.",
    }
    with pytest.raises(ValueError, match="SEC_MULTI_HYPOTHESIS_SOURCE_INVALID"):
        validate_ranked_interpretation(case, base)
