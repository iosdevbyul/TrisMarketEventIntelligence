from tris_market_event_intelligence.rule_critic import leakage_flags


def test_reject_outcome_conditioned_exception():
    rule = {
        "exceptions": (
            "Do not apply if the market reaction is positive or neutral "
            "despite the negative news."
        ),
    }
    assert leakage_flags(rule) == ["OUTCOME_CONDITIONED_EXCEPTION"]


def test_allow_preforecast_exception():
    rule = {"exceptions": "Do not apply if the article does not identify the company."}
    assert leakage_flags(rule) == []
