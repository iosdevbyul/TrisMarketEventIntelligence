from tris_market_event_intelligence.event_integration_relations import cross_source_edges, features

STAMP = "2026-11-13T12:00:00+00:00"


def article(kind, source, title, body="", **extra):
    return {"kind": kind, "source": source, "title": title, "captureVerified": True,
            "quality": "TEXT_CANDIDATE", "publisherDeclaredAt": STAMP,
            "declaredClockConsistent": True, "integrationFeatures": features(kind, source, title, body),
            **extra}


def test_explicit_period_and_series_not_same_time_only():
    release = article("ECONOMIC_RELEASE", "bls_cpi", "Consumer Price Index October 2026")
    news = article("NEWS", "bbc_business", "US CPI October 2026 rises")
    foreign = article("NEWS", "bbc_business", "UK CPI October 2026 rises")
    wrong_period = article("NEWS", "bbc_business", "US CPI September 2026 rises")
    forecast = article("NEWS", "bbc_business", "US CPI October 2026 forecast")
    obs = {"kind": "ECONOMIC_OBSERVATION", "source": "FRED_CSV", "series": "CPIAUCSL",
           "period": "2026-10-01", "quality": "NUMERIC_CONTEXT_ONLY"}
    edges = list(cross_source_edges([release, news, foreign, wrong_period, forecast, obs]))
    assert [e[2] for e in edges] == ["INDICATOR_REPORT_CANDIDATE", "RELEASE_SERIES_PERIOD_CONTEXT"]
    assert edges[0][1] == news
    assert not edges[1][3]["originalReleaseValueVerified"]


def test_accession_mention_not_generic_issuer():
    accession = "0001045810-26-000001"
    sec = article("SEC", "sec_filing_1045810", "NVIDIA SEC 8-K",
                  secIdentity=["1045810", accession])
    news = article("NEWS", "bbc_business", "NVIDIA report", "Filing accession " + accession)
    unrelated = article("NEWS", "bbc_business", "NVIDIA report")
    edges = list(cross_source_edges([sec, news, unrelated]))
    assert len(edges) == 1
    assert edges[0][2] == "SEC_ACCESSION_MENTION_CANDIDATE"


def test_ambiguous_period_and_non_us_news():
    assert not features("NEWS", "bbc_business", "UK CPI October 2026", "")["indicators"]
    assert not features("ECONOMIC_RELEASE", "bls_cpi",
                        "CPI September 2026 and October 2026", "")["indicators"]
    assert features("ECONOMIC_RELEASE", "bea_news", "GDP third quarter 2026", "")[
        "indicators"
    ][0]["period"] == "2026-07-01"
