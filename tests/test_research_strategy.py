import json

from tris_market_event_intelligence.experience_memory import digest
from tris_market_event_intelligence.research_strategy import (
    choose_strategy,
    fingerprint,
    rejected_history,
)


def test_rejected_hypothesis_survives_new_rule_id(tmp_path):
    rule = {
        "condition": "Leadership risk",
        "analysis_action": "Assess company exposure",
        "exceptions": "Missing original evidence",
        "hypothesis": "Governance risk affects valuation",
        "researchCutoff": "2026-11-26T12:00:00+00:00",
    }
    rule["ruleId"] = digest(rule)
    (tmp_path / (rule["ruleId"] + ".rule.json")).write_text(
        json.dumps(rule), encoding="utf-8",
    )
    review = {
        "originalRuleId": rule["ruleId"],
        "initialReview": {
            "effectiveVerdict": "REVISE",
            "reason": "Evidence insufficient",
            "deterministicFlags": [],
        },
        "revisedRule": None,
        "revisionReview": None,
    }
    (tmp_path / (digest(review) + ".review.json")).write_text(
        json.dumps(review), encoding="utf-8",
    )
    history = rejected_history(tmp_path)
    assert len(history) == 1
    later = {**rule, "researchCutoff": "2026-11-27T12:00:00+00:00"}
    assert history[0]["fingerprint"] == fingerprint(later)
    records = [{"outcome": {"articleId": i}} for i in range(3)]
    assert choose_strategy(history, records) == "EXPLORE_ALTERNATIVE_HYPOTHESIS"
    assert choose_strategy([], []) == "ACQUIRE_MORE_EXPERIENCES"
