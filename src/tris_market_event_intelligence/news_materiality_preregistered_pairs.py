"""Pre-register model-independent NEWS article pairs before human/model scoring.

Pair sampling uses pinned source chronology and (unverified) archived story
groups only. It does not inspect model output, labels, or protected TEST.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

from tris_market_event_intelligence.news_importance_review import _cohort, _instant, _read
from tris_market_event_intelligence.news_materiality_hypothesis import BANDS
from tris_market_event_intelligence.news_materiality_transition_review import (
    CHANGES,
    MIN_DESCRIPTIVE_PAIRS,
    RELATIONS,
    _pinned_pairs,
)
from tris_market_event_intelligence.news_materiality_review import _validation_ids

SCHEMA = "news_materiality_preregistered_pairs_v1"
SEED = "learning05-independent-news-pair-sampling-v1"
MAX_PAIRS = 100
MIN_EXPLORATORY_PAIRS = MIN_DESCRIPTIVE_PAIRS


def _ordered_pair(a: dict[str, object],
                  b: dict[str, object]) -> tuple[int, int]:
    """Use only first observed chronology; equal times break by source ID."""
    by_time = sorted((a, b), key=lambda x: (
        _instant(x["firstObservedAt"]), x["articleId"],
    ))
    return by_time[0]["articleId"], by_time[1]["articleId"]


def _registered_pairs(cohort: dict[str, object],
                      val: set[int]) -> list[dict[str, object]]:
    """Linear bounded deterministic pre-registration, no model or gold access."""
    articles = [a for a in cohort["items"] if a["articleId"] in val]
    if len(articles) < 2:
        return []
    key = hashlib.sha256(SEED.encode()).digest()
    def shuffled(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        return sorted(rows, key=lambda r: hashlib.sha256(
            key + str(r["articleId"]).encode() + r["textSha256"].encode()
        ).digest())

    # Declared story groups offer *unverified* candidate repeated-event
    # examples. This sampling deliberately also includes cross-group pairs.
    groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    for article in articles:
        group = article.get("storyGroupId")
        if type(group) is int and group > 0:
            groups[group].append(article)
    within: list[tuple[int, int]] = []
    for group in sorted(groups):
        if len(groups[group]) >= 2:
            within.extend(_ordered_pair(a, b) for a, b in pairwise(shuffled(groups[group])))
    cross = []
    rotated = shuffled(articles)
    for a, b in pairwise(rotated):
        if a.get("storyGroupId") is None or a.get("storyGroupId") != b.get("storyGroupId"):
            cross.append(_ordered_pair(a, b))
    # Allocate at most half to source-declared same-group candidates,
    # reserving the other half for potentially unrelated controls.
    chosen: list[dict[str, object]] = []
    seen: set[tuple[int, int]] = set()
    for label, candidates, quota in (
        ("DECLARED_STORY_GROUP_UNVERIFIED", within, MAX_PAIRS // 2),
        ("CROSS_GROUP_OR_UNKNOWN", cross, MAX_PAIRS),
    ):
        accepted = 0
        for pair in candidates:
            if len(chosen) >= MAX_PAIRS or accepted >= quota:
                break
            if pair in seen:
                continue
            seen.add(pair)
            chosen.append({
                "earlierArticleId": pair[0], "laterArticleId": pair[1],
                "samplingStratum": label,
            })
            accepted += 1
    return sorted(chosen, key=lambda p: (
        p["earlierArticleId"], p["laterArticleId"],
    ))


def prepare_preregistered_pairs(*, cohort_path: Path,
                                holdout_path: Path) -> dict[str, object]:
    """Create the blind pair registry without opening a model report."""
    raw, cohort = _cohort(cohort_path)
    val = _validation_ids(holdout_path, raw, cohort)
    pairs = _registered_pairs(cohort, val)
    return {
        "schemaVersion": SCHEMA,
        "samplingPolicy": SEED,
        "cohortSha256": hashlib.sha256(raw).hexdigest(),
        "holdoutSha256": hashlib.sha256(_read(holdout_path)).hexdigest(),
        "judgments": [{
            **p, "relation": None, "humanChange": None,
            "reviewer": None, "reviewedAt": None, "reason": None,
        } for p in pairs],
        "candidateSelectionIndependentOfModel": True,
        "modelMagnitudesExposedInTemplate": False,
        "humanGoldIndependentlyAudited": False,
        "generatedHumanLabels": 0,
        "stockPricesUsed": False,
    }


def evaluate_preregistered_pairs(*, cohort_path: Path, holdout_path: Path,
                                 review_path: Path,
                                 report_path: Path) -> dict[str, object]:
    """Score only genuinely human-reviewed same-event pairs in VALIDATION."""
    original = prepare_preregistered_pairs(
        cohort_path=cohort_path, holdout_path=holdout_path,
    )
    raw_review = _read(review_path)
    doc = json.loads(raw_review)
    if not isinstance(doc, dict) or set(doc) != set(original) or any(
        doc.get(k) != value for k, value in original.items() if k != "judgments"
    ):
        raise ValueError("Review not pinned to original model-independent pair registry")
    judges = doc.get("judgments")
    if not isinstance(judges, list) or len(judges) != len(original["judgments"]):
        raise ValueError("Review must preserve the entire pre-registered pair set")
    # Reuse the existing comprehensive research-report provenance replay.
    hashes, _model_selected_pairs_unused = _pinned_pairs(
        cohort_path=cohort_path, holdout_path=holdout_path,
        report_path=report_path,
    )
    report = json.loads(_read(report_path))
    hypotheses = {
        h["article_id"]: h["potential_magnitude"]
        for h in report["result"]["hypotheses"]
    }
    articles = {a["articleId"]: a for a in _cohort(cohort_path)[1]["items"]}
    expected = {
        (p["earlierArticleId"], p["laterArticleId"]): p
        for p in original["judgments"]
    }
    seen: set[tuple[int, int]] = set()
    counts: Counter[str] = Counter()
    scored = correct = ordinal_error = unassessable = 0
    unreviewed = unrelated = uncertain = model_abstained = 0
    for row in judges:
        if not isinstance(row, dict) or set(row) != set(original["judgments"][0] if (
            original["judgments"]) else {
                "earlierArticleId", "laterArticleId", "samplingStratum",
                "relation", "humanChange", "reviewer", "reviewedAt", "reason",
            }):
            raise ValueError("Invalid registered-pair review row")
        pair = (row["earlierArticleId"], row["laterArticleId"])
        if pair not in expected or pair in seen or (
            row["samplingStratum"] != expected[pair]["samplingStratum"]
        ):
            raise ValueError("Pair identities or sampling strata were changed")
        seen.add(pair)
        fields = [row[k] for k in (
            "relation", "humanChange", "reviewer", "reviewedAt", "reason",
        )]
        if all(v is None for v in fields):
            unreviewed += 1
            continue
        if row["relation"] not in RELATIONS or row["humanChange"] not in CHANGES:
            raise ValueError("Unsupported human relation or magnitude change")
        if row["relation"] != "SAME_EVENT" and row["humanChange"] != "UNASSESSABLE":
            raise ValueError("Unrelated or uncertain events cannot have materiality change")
        if not isinstance(row["reviewer"], str) or not 1 <= len(
            row["reviewer"].strip()
        ) <= 100 or not isinstance(row["reason"], str) or not 20 <= len(
            row["reason"].strip()
        ) <= 2000 or _instant(row["reviewedAt"]) < _instant(
            articles[pair[1]]["firstObservedAt"]
        ):
            raise ValueError("Timestamped human review with substantive reason required")
        counts[row["relation"]] += 1
        if row["relation"] == "DIFFERENT_EVENT":
            unrelated += 1
            continue
        if row["relation"] == "UNCERTAIN":
            uncertain += 1
            continue
        if row["humanChange"] == "UNASSESSABLE":
            unassessable += 1
            continue
        before, after = (hypotheses[pair[0]], hypotheses[pair[1]])
        if before not in BANDS or after not in BANDS:
            model_abstained += 1
            continue
        gap = BANDS.index(after) - BANDS.index(before)
        change = "INCREASE" if gap > 0 else "DECREASE" if gap < 0 else "UNCHANGED"
        scored += 1
        correct += change == row["humanChange"]
        ordinal = {"DECREASE": -1, "UNCHANGED": 0, "INCREASE": 1}
        ordinal_error += abs(ordinal[change] - ordinal[row["humanChange"]])
    return {
        "schemaVersion": "news_materiality_preregistered_pair_audit_v1",
        "status": ("DESCRIPTIVE_ONLY_NOT_INDEPENDENT_VALIDATION"
                   if scored >= MIN_EXPLORATORY_PAIRS
                   else "INSUFFICIENT_HUMAN_PAIR_GOLD"),
        "cohortSha256": original["cohortSha256"],
        "holdoutSha256": original["holdoutSha256"],
        "reviewSha256": hashlib.sha256(raw_review).hexdigest(),
        "reportSha256": hashes["reportSha256"],
        "samplingPolicy": SEED,
        "selectedValidationPairs": len(expected),
        "reviewedPairs": len(expected) - unreviewed,
        "unreviewedPairs": unreviewed,
        "differentEvents": unrelated, "uncertainEventIdentity": uncertain,
        "unassessableHumanChange": unassessable,
        "modelAbstained": model_abstained, "scoredSameEventPairs": scored,
        "sameEventDirectionAgreement": correct / scored if scored else None,
        "meanDirectionOrdinalError": ordinal_error / scored if scored else None,
        "humanRelationCounts": dict(sorted(counts.items())),
        "pairSelectionDidNotReadModelOutputs": True,
        "humanGoldIndependentlyAudited": False,
        "modelPerformanceImprovementValidated": False,
        "testGoldScored": False, "promotionAllowed": False,
        "stockPricesUsed": False, "productionDbModified": False,
    }
