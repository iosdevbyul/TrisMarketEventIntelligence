"""Tests for saved rule attempt diagnosis."""
import pytest

from tris_market_event_intelligence.market_research_diagnostics_cli import classify_attempt


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({"rule": {"ruleId": "R"}}, "CANDIDATE_ACCEPTED_FOR_RESEARCH"),
        ({"candidateCount": 1, "review": {"initialReview": {}}},
         "CRITIC_DID_NOT_ACCEPT"),
        ({"candidateCount": 1}, "CRITIC_RECORD_MISSING"),
        ({"errors": []}, "EMPTY_CANDIDATE_RESPONSE"),
        ({"errors": [{"reason": "INSUFFICIENT_DISTINCT_MIXED_EXPERIENCES"}]},
         "INSUFFICIENT_DISTINCT_MIXED_EXPERIENCES"),
        ({"errors": [{"reason": "Invalid experience alias"}]},
         "VALIDATION_ERRORS_WITH_NO_FINAL_CANDIDATE"),
    ],
)
def test_classify_attempt(record, expected):
    assert classify_attempt(record) == expected
