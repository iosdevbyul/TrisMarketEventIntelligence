from tris_market_event_intelligence.event_timeline_bridge_cli import (
    attach_outcomes,
    event_dates,
    reaction_targets,
)


def test_action_date_and_reaction_are_separate():
    text = "On November 22, 2026, the Board reinstated the CEO."
    fact = {"evidence": {"text": text}}
    assert event_dates({"kind": "SEC"}, fact)[0]["date"] == "2026-11-22"
    news = {"documents": [{"text": "두산(000150)은 전 거래일보다 13.49% 상승한 가격에 거래되고 있다.",
                           "bodySha256": "b", "rawSha256": "r", "observedAt": "t"}]}
    reactions, targets = reaction_targets(news)
    assert reactions
    assert targets[0]["symbolCandidate"] == "000150"
    assert not targets[0]["mappingVerified"]


def test_outcomes_attach_only_to_exact_article_and_body():
    records = [{"sources": [{"eventId": "article:1", "evidence": {"bodySha256": "a"}}]}]
    assert attach_outcomes(records, {"rows": [
        {"articleId": 1, "bodySha256": "wrong"},
        {"articleId": 1, "bodySha256": "a", "status": "OBSERVED_UNAUTHENTICATED_PRECOMMITMENT"},
    ]}) == 1
    assert records[0]["outcomeAttribution"] == "ORIGINAL_ARTICLE_ONLY_NOT_EVENT_CAUSALITY"
