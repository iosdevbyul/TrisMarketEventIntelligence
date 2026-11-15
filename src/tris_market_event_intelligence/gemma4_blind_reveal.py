"""Unblind complete human NEWS reviews only after exact input and mapping audit.

This is a descriptive pilot aggregate, never verified independent human gold
and never a production model promotion or measured accuracy improvement.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations

from tris_market_event_intelligence.gemma4_blind_review import (
    KEY_SCHEMA,
    PACKET_SCHEMA,
    RATINGS,
    _hash,
    audit_blind_judgments,
)
from tris_market_event_intelligence.gemma4_news_comparison import _validate_models

DIMENSIONS = ("evidenceAccuracy", "contextAndMechanism", "uncertaintyHonesty")
PREFERENCES = ("A", "B", "TIE", "BOTH_INADEQUATE", "UNDECIDED")
MAX_REVIEWERS = 10


def _mapping(
    packet: dict[str, object], mapping: dict[str, object],
) -> tuple[str, str]:
    if not isinstance(packet, dict) or packet.get("schemaVersion") != PACKET_SCHEMA:
        raise ValueError("Original blind packet required")
    if not isinstance(mapping, dict) or (
        mapping.get("schemaVersion") != KEY_SCHEMA
        or mapping.get("packetSha256") != _hash(packet)
        or mapping.get("revealOnlyAfterBlindReview") is not True
        or mapping.get("promotionAllowed") is not False
        or set(mapping) != {
            "schemaVersion", "packetSha256", "assignments",
            "revealOnlyAfterBlindReview", "promotionAllowed",
        }
    ):
        raise ValueError("Sealed model mapping does not match original packet")
    original = packet.get("items")
    assigned = mapping.get("assignments")
    if not isinstance(original, list) or not original or not isinstance(
        assigned, list,
    ) or len(assigned) != len(original):
        raise ValueError("Incomplete sealed assignment list")
    models: set[str] = set()
    seen: set[int] = set()
    for source, binding in zip(original, assigned, strict=True):
        if not isinstance(source, dict) or not isinstance(binding, dict) or (
            set(binding) != {"eventId", "newsInputSha256", "A", "B"}
            or type(binding["eventId"]) is not int
            or binding["eventId"] in seen
            or binding["eventId"] != source.get("eventId")
            or binding["newsInputSha256"] != source.get("newsInputSha256")
            or binding["A"] == binding["B"]
        ):
            raise ValueError("Sealed model assignment invalid or misaligned")
        seen.add(binding["eventId"])
        models.update((binding["A"], binding["B"]))
    if len(models) != 2 or "gemma2:latest" not in models:
        raise ValueError("Sealed assignment must contain two pinned model identities")
    candidate = next(name for name in models if name != "gemma2:latest")
    _validate_models(("gemma2:latest", candidate))
    for binding in assigned:
        if {binding["A"], binding["B"]} != models:
            raise ValueError("Event assignments disagree on compared models")
    return "gemma2:latest", candidate


def _agreement(
    ratings: list[dict[int, str]],
) -> dict[str, object]:
    if len(ratings) < 2:
        return {
            "status": "NOT_ESTIMABLE_SINGLE_REVIEWER",
            "reviewerPairs": 0,
            "itemsCompared": 0,
            "exactAgreementRate": None,
            "cohensKappa": None,
        }
    paired = []
    kappa_values = []
    for first, second in combinations(ratings, 2):
        if set(first) != set(second):
            raise ValueError("Reviewers did not grade the same events")
        ids = sorted(first)
        agrees = sum(first[key] == second[key] for key in ids)
        paired.append((agrees, len(ids)))
        marg_a = Counter(first.values())
        marg_b = Counter(second.values())
        expected = sum(
            (marg_a[label] / len(ids)) * (marg_b[label] / len(ids))
            for label in PREFERENCES
        )
        if expected < 1:
            kappa_values.append((agrees / len(ids) - expected) / (1 - expected))
    compared = sum(n for _, n in paired)
    return {
        "status": "DESCRIPTIVE_PAIRWISE_ONLY",
        "reviewerPairs": len(paired),
        "itemsCompared": compared,
        "exactAgreementRate": round(
            sum(matches for matches, _ in paired) / compared, 6,
        ),
        "cohensKappa": (
            round(sum(kappa_values) / len(kappa_values), 6)
            if len(ratings) == 2 and kappa_values else None
        ),
        "kappaUnavailableReason": (
            "DEGENERATE_MARGINALS" if len(ratings) == 2 and not kappa_values
            else "PAIRWISE_KAPPA_NOT_POOLED" if len(ratings) > 2 else None
        ),
    }


def reveal_blind_review(
    *, packet: dict[str, object], mapping: dict[str, object],
    reviews: list[tuple[str, dict[str, object]]],
) -> dict[str, object]:
    """Validate all reviews before inspecting or aggregating sealed model IDs.

    Reviewer identity is user-supplied, not independently authenticated.
    Never include model output text, reviewer reasons or human gold in report.
    """
    if not isinstance(reviews, list) or not 1 <= len(reviews) <= MAX_REVIEWERS:
        raise ValueError("Provide 1..10 completed human review documents")
    ids: set[str] = set()
    review_hashes: list[dict[str, str]] = []
    for reviewer, document in reviews:
        if not isinstance(reviewer, str) or reviewer.strip() in ids:
            raise ValueError("Each review requires a unique human reviewer ID")
        audited = audit_blind_judgments(packet, document, reviewer=reviewer)
        ids.add(reviewer.strip())
        review_hashes.append({
            "reviewer": reviewer.strip(),
            "reviewSha256": audited["reviewSha256"],
        })
    # Do not open the sealed mapping unless every review is fully validated.
    base, challenger = _mapping(packet, mapping)
    totals: dict[str, dict[str, Counter[str]]] = {
        model: {dim: Counter() for dim in DIMENSIONS}
        for model in (base, challenger)
    }
    preference_totals: Counter[str] = Counter()
    by_reviewer = []
    mapped = mapping["assignments"]
    preference_rows: list[dict[int, str]] = []
    for (reviewer, document), fingerprint in zip(
        reviews, review_hashes, strict=True,
    ):
        local: Counter[str] = Counter()
        preferences: dict[int, str] = {}
        for choice, binding in zip(document["items"], mapped, strict=True):
            grade = choice["humanReview"]
            verdict = grade["preferredResponse"]
            preferences[binding["eventId"]] = verdict
            if verdict in {"A", "B"}:
                selected = binding[verdict]
                local[selected] += 1
                preference_totals[selected] += 1
            else:
                local[verdict] += 1
                preference_totals[verdict] += 1
            for slot in ("A", "B"):
                model = binding[slot]
                for dimension in DIMENSIONS:
                    rating = grade[slot][dimension]
                    if rating not in RATINGS:
                        raise ValueError("Review rating not in original rubric")
                    totals[model][dimension][rating] += 1
        preference_rows.append(preferences)
        by_reviewer.append({
            **fingerprint,
            "pairedNewsReviewed": len(mapped),
            "preferences": dict(sorted(local.items())),
        })
    return {
        "schemaVersion": "gemma4_blind_reveal_summary_v1",
        "status": "HUMAN_REVIEW_REVEALED_DESCRIPTIVE_ONLY",
        "packetSha256": _hash(packet),
        "mappingSha256": _hash(mapping),
        "reviewers": by_reviewer,
        "reviewerCount": len(reviews),
        "pairedNewsReviewed": len(mapped),
        "modelNames": [base, challenger],
        "modelPreferences": dict(sorted(preference_totals.items())),
        "modelRubricGradeCounts": {
            name: {
                dim: dict(sorted(totals[name][dim].items()))
                for dim in DIMENSIONS
            } for name in (base, challenger)
        },
        "interReviewerPreferenceAgreement": _agreement(preference_rows),
        "independentReviewerIdentityVerified": False,
        "independentSourceFactCheckVerified": False,
        "smallSamplePilot": True,
        "improvementProven": False,
        "protectedTestGoldUsed": False,
        "modelPromotionAllowed": False,
        "productionModelChanged": False,
        "stockPricesUsed": False,
    }
