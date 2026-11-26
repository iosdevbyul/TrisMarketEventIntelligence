from tris_market_event_intelligence.cross_source_match_cli import discover, metric_facts


def item(kind, source, key, title, body):
    return {"kind": kind, "source": source, "eventId": key, "title": title,
            "url": "https://example.org/" + key, "claimKind": "FACT",
            "publisherDeclaredAt": "2026-11-25T12:00:00+00:00",
            "secAcceptanceAt": None, "secForm": None, "issuers": [],
            "documents": [{"text": body, "bodySha256": key, "rawSha256": key,
                           "observedAt": "2026-11-25T13:00:00+00:00"}]}


def test_bilingual_metric_match_requires_comparable_value_period_and_basis():
    official = item("ECONOMIC_RELEASE", "bea_news", "a", "PCE October 2026",
                    "The PCE price index increased 3.8 percent from a year ago in October 2026.")
    news = item("NEWS", "paper", "b", "미국 10월 PCE",
                "미국 10월 PCE 가격지수는 전년 동월 대비 3.8% 상승했다.")
    pairs, _ = discover([official, news])
    assert len(pairs) == 1
    assert pairs[0]["key"][:5] == ["US", "PCE", "HEADLINE", "2026-10", "YOY"]
    assert not pairs[0]["predictionEligible"]
    for body in (
        "미국 10월 근원 PCE 가격지수는 전년 동월 대비 3.8% 상승했다.",
        "미국 10월 PCE 가격지수는 전월 대비 3.8% 상승했다.",
        "미국 9월 PCE 가격지수는 전년 동월 대비 3.8% 상승했다.",
        "미국 10월 PCE 가격지수 예상치는 전년 동월 대비 3.8% 상승이다.",
        "미국 10월 PCE 가격지수는 전년 동월 대비 3.8% 하락했다.",
    ):
        changed = {**news, "documents": [{**news["documents"][0], "text": body}]}
        assert not discover([official, changed])[0]


def test_missing_period_and_multiple_metrics_abstain():
    news = item("NEWS", "paper", "b", "미국 물가",
                "미국 PCE 가격지수는 전년 동월 대비 3.8% 상승했다.")
    assert not metric_facts(news)


def test_same_original_not_cross_source_positive():
    official = item("ECONOMIC_RELEASE", "bea_news", "a", "PCE October 2026",
                    "The PCE price index increased 3.8 percent from a year ago in October 2026.")
    news = {**official, "kind": "NEWS", "source": "paper", "eventId": "b",
            "title": "US PCE October 2026"}
    assert not discover([official, news])[0]
