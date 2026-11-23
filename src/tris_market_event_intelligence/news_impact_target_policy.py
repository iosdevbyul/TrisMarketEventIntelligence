"""Strictly scoped structural guardrails for market-impact *targets*.

A central bank, policy instrument, economic variable, or monetary operation
is not an investable market target. This is a deliberately narrow rule set,
not a general named-entity recognizer or a factual/causal correctness test.
"""
from __future__ import annotations

import re

_NON_MARKET_LABELS = frozenset({
    "federal reserve", "federal reserve board", "board of governors",
    "federal open market committee", "fomc", "federal funds rate",
    "federal funds target rate", "target federal funds rate",
    "primary credit rate", "discount rate", "policy rate",
    "federal funds rate / primary credit rate",
    "open market operations", "repurchase agreement operations",
    "standing repurchase agreement operations", "reverse repurchase operations",
    "economic activity", "inflation", "consumer spending", "monetary policy",
    "treasury yields", "longer-maturity treasury yields",
    "yield curve", "unemployment rate", "interest rates",
})

# These aliases refer to the same broad-market category, but not necessarily
# to a stock index. Keep the original model target visible in saved analyses.
_UNAMBIGUOUS_ALIASES = {
    "financial markets": "financial_markets",
    "general market": "financial_markets",
    "broad financial market": "financial_markets",
    "overall financial markets": "financial_markets",
}


def normalized_label(target: str) -> str:
    return re.sub(r"\s+", " ", target.casefold()).strip()


def target_key(target: str) -> str:
    label = normalized_label(target)
    return _UNAMBIGUOUS_ALIASES.get(label, label)


def target_error(target: str) -> str | None:
    """Reject known non-investable target *labels*, not their factual mentions."""
    label = normalized_label(target)
    if label in _NON_MARKET_LABELS:
        return "NON_MARKET_TARGET"
    if not label or len(label) > 180:
        return "TARGET_LABEL_INVALID"
    return None
