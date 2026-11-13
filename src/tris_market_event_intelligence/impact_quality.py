"""Conservative, deterministic screening of obviously unfinished impact rationales.

This is not a factuality, causal-validity, or market-direction verifier.
Do not rewrite model prose to satisfy the checks.
"""

from __future__ import annotations

import re

# Intentionally narrow: common dangling connectives observed in local output.
# Avoid a blanket Korean "-고" rule, which could reject complete nouns.
_DANGLING_KOREAN = re.compile(
    r"(?:지속되고|증가하고|감소하고|확대되고|축소되고|"
    r"이어지고|나타나고|보이고|높아지고|낮아지고|"
    r"되고|있고|했으며|하며|이지만|그러나|그리고|반면)$"
)
_DANGLING_ENGLISH = re.compile(
    r"\b(?:and|or|but|because|although|whereas|while|if|"
    r"which|that|with|from|into|of|for|to|on)\s*$",
    re.IGNORECASE,
)


def rationale_quality_code(rationale: str) -> str | None:
    """Return a safe failure code only for a high-confidence unfinished ending."""
    text = rationale.strip()
    if not text:
        return "RATIONALE_EMPTY"
    # A trailing clause connector remains unfinished even if punctuation follows.
    tail = text.rstrip(" \t\r\n.,;:!?…")
    if _DANGLING_KOREAN.search(tail) or _DANGLING_ENGLISH.search(tail):
        return "RATIONALE_INCOMPLETE"
    return None
