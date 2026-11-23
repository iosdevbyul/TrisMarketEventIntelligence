"""Conservative repeat-run agreement gate for source-backed news interpretations.

The gate does not normalize away model disagreement, resolve truth, or use prices.
Matching output remains unreviewed; disagreement requires human source review.
"""
from __future__ import annotations


def _signature(analysis: dict) -> list[tuple[str, str]]:
    return sorted(
        (impact["target"].strip().casefold(), impact["direction"])
        for impact in analysis["impacts"]
    )


def compare_analyses(first: dict, second: dict) -> dict:
    """Compare complete target/direction sets while preserving both originals."""
    first_signature = _signature(first)
    second_signature = _signature(second)
    first_map = {name: direction for name, direction in first_signature}
    second_map = {name: direction for name, direction in second_signature}
    common = sorted(set(first_map) & set(second_map))
    conflicting = [
        {"target": target, "first": first_map[target], "second": second_map[target]}
        for target in common if first_map[target] != second_map[target]
    ]
    exact = first_signature == second_signature
    return {
        "status": "REPEATABLE_UNREVIEWED" if exact else "REVIEW_REQUIRED",
        "runs": 2,
        "exactTargetDirections": exact,
        "firstImpactCount": len(first_signature),
        "secondImpactCount": len(second_signature),
        "commonTargets": common,
        "firstOnlyTargets": sorted(set(first_map) - set(second_map)),
        "secondOnlyTargets": sorted(set(second_map) - set(first_map)),
        "directionConflicts": conflicting,
        "secondAnalysis": second,
        "semanticCorrectnessVerified": False,
        "priceDataUsed": False,
    }
