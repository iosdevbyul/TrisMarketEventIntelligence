"""Regression tests for sequential, resumable market-learning loop."""
from __future__ import annotations

import json
import sqlite3
from argparse import Namespace
from datetime import UTC, datetime, timedelta

import pytest

from tris_market_event_intelligence import sequential_market_learning_cli as seq


def test_prediction_target_validation():
    prediction = seq.canonical_prediction({
        "summary": "An event",
        "impacts": [
            {"targetType": "STOCK", "target": "NVDA", "direction": "NEGATIVE",
             "reason": "Supply restrictions"},
            {"targetType": "SECTOR", "target": "SEMICONDUCTORS",
             "direction": "NEGATIVE", "reason": "Demand"},
        ],
    })
    assert prediction["impacts"][1]["priceProxy"] == "SMH"
    with pytest.raises(ValueError, match="Unsupported"):
        seq.canonical_prediction({"impacts": [{
            "targetType": "SECTOR", "target": "NOT_A_SECTOR",
            "direction": "POSITIVE", "reason": "Unknown",
        }]})


class ScriptedModel:
    def __init__(self):
        self.prompts = []
        self.predictions = 0

    def generate_json(self, *, system, user, schema):
        payload = json.loads(user)
        self.prompts.append(payload)
        if "wrongOutcomes" in payload:
            return json.dumps({
                "diagnosis": "The negative scenario overlooked strong demand",
                "lesson": "When production is constrained, check whether demand "
                          "still supports pricing power before predicting a decline.",
            })
        self.predictions += 1
        return json.dumps({
            "summary": "Supply-side headline",
            "impacts": [
                {"targetType": "STOCK", "target": "NVDA",
                 "direction": "NEGATIVE" if self.predictions == 1 else "POSITIVE",
                 "reason": "Supply pressures"},
            ],
        })


def sample(article_id, date):
    return {
        "articleId": article_id, "source": "test",
        "title": "Chip manufacturing supply event",
        "summary": "A reported production shift could influence supplier revenue",
        "firstObservedAt": date.isoformat(),
        "publishedAtReported": date.isoformat(),
    }


def args(tmp_path):
    return Namespace(
        source_db=tmp_path / "news.sqlite3",
        journal=tmp_path / "learning.sqlite3",
        as_of="2026-11-19T00:00:00+00:00",
        count=2, horizon_hours=24, interval="1d", model="gemma4:e2b",
    )


def test_wrong_answer_updates_next_article_and_resume(monkeypatch, tmp_path):
    first = datetime(2025, 1, 1, tzinfo=UTC)
    articles = [sample(10, first), sample(11, first + timedelta(days=7))]
    monkeypatch.setattr(seq, "load_articles",
                        lambda db, count, cutoff: (articles, 1))

    def fake_price(client, symbol, observed, horizon, interval):
        assert symbol in {"NVDA", "SPY"}
        return {
            "return": 0.03 if symbol == "NVDA" else 0.01,
            "baselineAt": (observed - timedelta(hours=1)).isoformat(),
            "outcomeAt": (observed + timedelta(days=2)).isoformat(),
        }

    monkeypatch.setattr(seq, "price_return", fake_price)
    fake = ScriptedModel()
    config = args(tmp_path)
    result = seq.run(config, model=fake, client=object())
    assert result["articlesCompleted"] == 2
    assert result["lessons"] == 1
    # Two primary predictions plus one pre-outcome baseline comparison.
    assert fake.predictions == 3
    assert fake.prompts[0]["priorFeedbackLessons"] == []
    assert "pricing power" in fake.prompts[2]["priorFeedbackLessons"][0]
    with sqlite3.connect(config.journal) as db:
        predictions = db.execute(
            "SELECT prediction_json, phase FROM steps ORDER BY seq"
        ).fetchall()
        assert json.loads(predictions[0][0])["impacts"][0]["direction"] == "NEGATIVE"
        assert json.loads(predictions[1][0])["impacts"][0]["direction"] == "POSITIVE"
        assert [r[1] for r in predictions] == ["COMPLETE", "COMPLETE"]
    resumed = ScriptedModel()
    seq.run(config, model=resumed, client=object())
    assert resumed.prompts == []
    # Increasing a 2-case journal does not replay already evaluated articles.
    config.count = 3
    expanded = seq.run(config, model=resumed, client=object())
    assert expanded["articlesCompleted"] == 2
    assert expanded["status"] == "INSUFFICIENT_ELIGIBLE_ARTICLES"
    assert resumed.prompts == []


def test_no_feedback_on_unavailable_price(monkeypatch, tmp_path):
    articles = [sample(1, datetime(2025, 1, 1, tzinfo=UTC))]
    monkeypatch.setattr(seq, "load_articles",
                        lambda db, count, cutoff: (articles, 0))
    def missing(*args, **kwargs):
        raise ValueError("No historical bars")

    monkeypatch.setattr(seq, "price_return", missing)
    config = args(tmp_path)
    config.count = 1
    fake = ScriptedModel()
    result = seq.run(config, model=fake, client=object())
    assert result["marketReturnsAnalyzed"] is False
    assert result["lessons"] == 0
    assert len(fake.prompts) == 1


def test_retrospective_learning_continues_across_overlapping_articles(
    monkeypatch, tmp_path,
):
    first = datetime(2025, 1, 1, tzinfo=UTC)
    articles = [sample(10, first), sample(11, first + timedelta(hours=1))]
    monkeypatch.setattr(
        seq, "load_articles", lambda db, count, cutoff: (articles, 0),
    )

    def future_price(client, symbol, observed, horizon, interval):
        return {
            "return": 0.05,
            "baselineAt": (observed - timedelta(days=1)).isoformat(),
            "outcomeAt": (observed + timedelta(days=7)).isoformat(),
        }

    monkeypatch.setattr(seq, "price_return", future_price)
    fake = ScriptedModel()
    result = seq.run(args(tmp_path), model=fake, client=object())
    assert result["articlesCompleted"] == 2
    assert result["pointInTimeCertified"] is False
    assert result["evaluationMode"] == "RETROSPECTIVE_SEQUENTIAL_FEEDBACK"
    assert "pricing power" in fake.prompts[2]["priorFeedbackLessons"][0]


def test_same_day_articles_are_not_dropped_by_selection(monkeypatch, tmp_path):
    first = datetime(2026, 11, 1, tzinfo=UTC)
    entries = [sample(i + 1, first + timedelta(minutes=i)) for i in range(20)]
    for row in entries:
        row["textSha256"] = str(row["articleId"])
        row["publishedAtReported"] = row["firstObservedAt"]
    monkeypatch.setattr(
        seq, "historical_news_cohort",
        lambda **kwargs: {"items": entries},
    )
    selected, holdout = seq.load_articles(
        tmp_path / "source.sqlite3", 10, first + timedelta(days=1),
    )
    assert len(selected) == 10
    assert len({a["articleId"] for a in selected}) == 10
    assert holdout == 4
    assert [a["articleId"] for a in selected] == list(range(1, 11))


def test_gemma_case_normalization_and_retry():
    class InconsistentModel:
        def __init__(self):
            self.calls = 0

        def generate_json(self, *, system, user, schema):
            self.calls += 1
            if self.calls == 1:
                return json.dumps({"impacts": [{
                    "targetType": "COMPANY", "target": "NVDA",
                    "direction": "MIXED", "reason": "Conflicting signals",
                }]})
            return json.dumps({"impacts": [{
                "targetType": "company", "target": "nvda",
                "direction": "bearish", "reason": "Reduced export exposure",
            }]})

    model = InconsistentModel()
    answer = seq.predict_with_repair(model, {"article": {"title": "Export limits"}})
    assert model.calls == 2
    assert answer["impacts"][0]["direction"] == "NEGATIVE"
    assert answer["impacts"][0]["targetType"] == "STOCK"
    assert answer["impacts"][0]["priceProxy"] == "NVDA"


def test_invalid_model_output_does_not_become_prediction():
    class AlwaysInvalid:
        def generate_json(self, *, system, user, schema):
            return json.dumps({"impacts": [{
                "targetType": "STOCK", "target": "NVDA",
                "direction": "MIXED", "reason": "Undetermined",
            }]})

    with pytest.raises(ValueError, match="AFTER_3_ATTEMPTS"):
        seq.predict_with_repair(AlwaysInvalid(), {"article": {}})


def test_gemma_sector_name_in_target_type_is_normalized():
    """Regression for the real FINANCIALS / NEUTRAL output."""
    for target in (None, "financial services", "XLF", "FINANCIALS"):
        impact = {
            "targetType": "FINANCIALS",
            "direction": "NEUTRAL",
            "reason": "Rates offset demand",
        }
        if target is not None:
            impact["target"] = target
        output = seq.canonical_prediction({"impacts": [impact]})
        assert output["impacts"] == [{
            "targetType": "SECTOR", "target": "FINANCIALS", "priceProxy": "XLF",
            "direction": "NEUTRAL", "reason": "Rates offset demand",
        }]


def test_broad_market_name_in_target_type_is_normalized():
    prediction = seq.canonical_prediction({"impacts": [{
        "targetType": "BROAD_MARKET",
        "direction": "NEGATIVE", "reason": "Broad market risk",
    }]})
    assert prediction["impacts"][0]["targetType"] == "MARKET"
    assert prediction["impacts"][0]["priceProxy"] == "SPY"


def test_prior_market_context_excludes_future_news_and_prices():
    observed = datetime(2026, 11, 15, 10, tzinfo=UTC)
    news = sample(10, observed)
    older = sample(9, observed - timedelta(days=1))
    older["title"] = "Chip manufacturing supply event earlier"
    future = sample(11, observed + timedelta(hours=2))
    class FakeClient:
        def get(self, url, params, headers):
            assert params["period2"] == int(observed.timestamp())
            class Response:
                def raise_for_status(self):
                    return None
                def json(self):
                    return {"chart": {"result": [{
                        "timestamp": [
                            int((observed - timedelta(days=3)).timestamp()),
                            int((observed - timedelta(days=2)).timestamp()),
                            int((observed + timedelta(days=2)).timestamp()),
                        ],
                        "indicators": {"quote": [{"close": [100, 102, 999]}]},
                    }]}}
            return Response()
    context = seq.pre_event_market_context(news, [older, future], FakeClient())
    assert [x["articleId"] for x in context["earlierRelatedArticles"]] == [9]
    assert context["marketPreTrend"]["status"] == "AVAILABLE"
    assert context["marketPreTrend"]["return"] == pytest.approx(0.02)
    assert context["consensus"]["status"] == "UNKNOWN"


def test_market_context_missing_source_is_unknown():
    article = sample(1, datetime(2025, 1, 1, tzinfo=UTC))
    context = seq.pre_event_market_context(article, [], object())
    assert context["marketPreTrend"]["status"] == "UNAVAILABLE"
    assert context["consensus"]["status"] == "UNKNOWN"
    assert context["earlierRelatedArticles"] == []


def test_unsubstantiated_consensus_surprise_is_rejected():
    class Unsubstantiated:
        def generate_json(self, *, system, user, schema):
            return json.dumps({
                "summary": "Positive fundamentals",
                "impacts": [],
                "surpriseAssessment": {
                    "economicImpact": "POSITIVE",
                    "expectationStatus": "ABOVE",
                    "novelty": "NEW",
                    "pricedInStatus": "UNKNOWN",
                    "rationale": "Invented consensus",
                },
            })
    with pytest.raises(ValueError, match="AFTER_3_ATTEMPTS"):
        seq.predict_with_repair(Unsubstantiated(), {
            "marketContext": {
                "consensus": {"status": "UNKNOWN"},
                "earlierRelatedArticles": [],
            },
        })


def test_legacy_prediction_without_surprise_is_explicitly_unknown():
    model = ScriptedModel()
    prediction = seq.predict_with_repair(model, {
        "marketContext": {
            "consensus": {"status": "UNKNOWN"},
            "earlierRelatedArticles": [],
        },
    })
    assert prediction["surpriseAssessment"]["expectationStatus"] == "UNKNOWN"
    assert prediction["surpriseAssessment"]["pricedInStatus"] == "UNKNOWN"


def test_sector_etf_proxy_is_normalized_without_guessing():
    output = seq.canonical_prediction({"impacts": [
        {"targetType": "SECTOR", "target": "SMH",
         "direction": "NEGATIVE", "reason": "Supply limits"},
        {"targetType": "MARKET", "target": "SPY",
         "direction": "NEUTRAL", "reason": "Offsetting effects"},
    ]})
    assert output["impacts"][0]["target"] == "SEMICONDUCTORS"
    assert output["impacts"][0]["priceProxy"] == "SMH"
    assert output["impacts"][1]["target"] == "BROAD_MARKET"


def test_unrecognized_sector_is_rejected_without_aborting_other_targets():
    class InvalidSectorModel:
        def __init__(self):
            self.calls = 0

        def generate_json(self, *, system, user, schema):
            self.calls += 1
            return json.dumps({
                "summary": "Positive news",
                "impacts": [
                    {"targetType": "SECTOR", "target": "UNLISTED_INDUSTRY",
                     "direction": "POSITIVE", "reason": "New order"},
                    {"targetType": "STOCK", "target": "NVDA",
                     "direction": "POSITIVE", "reason": "Direct exposure"},
                ],
            })
    model = InvalidSectorModel()
    output = seq.predict_with_repair(model, {
        "marketContext": {
            "consensus": {"status": "UNKNOWN"},
            "earlierRelatedArticles": [],
        },
    })
    assert model.calls == 3
    assert [i["target"] for i in output["impacts"]] == ["NVDA"]
    assert output["recoveryStatus"] == "PARTIAL_TARGET_MAPPING"
    assert output["targetMappingRejections"][0]["target"] == "UNLISTED_INDUSTRY"
    assert output["surpriseAssessment"]["expectationStatus"] == "UNKNOWN"


def test_every_unsupported_target_is_abstention_not_a_false_hit():
    class UnsupportedOnly:
        def generate_json(self, *, system, user, schema):
            return json.dumps({"summary": "unclear", "impacts": [
                {"targetType": "SECTOR", "target": "UNSUPPORTED",
                 "direction": "POSITIVE", "reason": "Unknown exposure"},
            ]})
    output = seq.predict_with_repair(UnsupportedOnly(), {
        "marketContext": {"consensus": {"status": "UNKNOWN"},
                          "earlierRelatedArticles": []},
    })
    assert output["impacts"] == []
    assert output["recoveryStatus"] == "NO_VALID_TARGETS"
    assert len(output["targetMappingRejections"]) == 1


def test_invalid_direction_remains_hard_error_after_retries():
    class InvalidDirection:
        def generate_json(self, *, system, user, schema):
            return json.dumps({"summary": "unclear", "impacts": [
                {"targetType": "SECTOR", "target": "BAD_SECTOR",
                 "direction": "SIDEWAYS", "reason": "Unknown"},
            ]})
    with pytest.raises(ValueError, match="AFTER_3_ATTEMPTS"):
        seq.predict_with_repair(InvalidDirection(), {"article": {}})


def test_malformed_ticker_recovery_keeps_valid_stock_and_sector():
    class MalformedTickerModel:
        def __init__(self):
            self.calls = 0

        def generate_json(self, *, system, user, schema):
            self.calls += 1
            return json.dumps({
                "summary": "Company and sector impact",
                "impacts": [
                    {"targetType": "STOCK", "target": "NVIDIA CORP",
                     "direction": "NEGATIVE", "reason": "Export controls"},
                    {"targetType": "STOCK", "target": "AMD",
                     "direction": "NEGATIVE", "reason": "Export controls"},
                    {"targetType": "SECTOR", "target": "SEMICONDUCTORS",
                     "direction": "NEGATIVE", "reason": "Sector exposure"},
                ],
            })
    model = MalformedTickerModel()
    answer = seq.predict_with_repair(model, {
        "marketContext": {
            "consensus": {"status": "UNKNOWN"},
            "earlierRelatedArticles": [],
        },
    })
    assert model.calls == 3
    assert [(x["targetType"], x["target"]) for x in answer["impacts"]] == [
        ("STOCK", "AMD"), ("SECTOR", "SEMICONDUCTORS"),
    ]
    assert answer["recoveryStatus"] == "PARTIAL_TARGET_MAPPING"
    assert answer["targetMappingRejections"] == [{
        "targetType": "STOCK", "target": "NVIDIA CORP",
        "reasonCode": "MALFORMED_STOCK_TICKER",
    }]


def test_malformed_only_ticker_abstains_without_fake_success():
    class BadTickerModel:
        def generate_json(self, *, system, user, schema):
            return json.dumps({"summary": "unresolved", "impacts": [
                {"targetType": "STOCK", "target": "A/B",
                 "direction": "POSITIVE", "reason": "Possible upside"},
            ]})
    result = seq.predict_with_repair(BadTickerModel(), {
        "marketContext": {
            "consensus": {"status": "UNKNOWN"},
            "earlierRelatedArticles": [],
        },
    })
    assert result["impacts"] == []
    assert result["recoveryStatus"] == "NO_VALID_TARGETS"
    assert result["targetMappingRejections"][0]["reasonCode"] == "MALFORMED_STOCK_TICKER"


def test_malformed_ticker_recovery_rejects_other_invalid_fields():
    class MixedInvalidModel:
        def generate_json(self, *, system, user, schema):
            return json.dumps({"summary": "invalid", "impacts": [
                {"targetType": "STOCK", "target": "A B",
                 "direction": "POSITIVE", "reason": "Market impact"},
                {"targetType": "STOCK", "target": "AMD",
                 "direction": "UNKNOWN", "reason": "Unknown effect"},
            ]})
    with pytest.raises(ValueError, match="Invalid target type/direction"):
        seq.predict_with_repair(MixedInvalidModel(), {
            "marketContext": {
                "consensus": {"status": "UNKNOWN"},
                "earlierRelatedArticles": [],
            },
        })


def test_historical_article_uses_publication_time_not_ingestion(monkeypatch, tmp_path):
    published = datetime(2026, 5, 26, 18, tzinfo=UTC)
    ingested = datetime(2026, 11, 13, 20, tzinfo=UTC)
    rows = [sample(i, published + timedelta(days=i)) for i in range(1, 11)]
    for row in rows:
        row["firstObservedAt"] = ingested.isoformat()
    monkeypatch.setattr(seq, "historical_news_cohort",
                        lambda **kwargs: {"items": list(reversed(rows))})
    articles, reserved = seq.load_articles(
        tmp_path / "news.sqlite3", 8, ingested + timedelta(days=1),
    )
    assert len(articles) == 8
    assert reserved == 2
    assert [r["articleId"] for r in articles] == list(range(1, 9))
    assert seq.publication_time(articles[0]) < ingested


def test_reaction_price_uses_first_release_day_close_and_prior_close(monkeypatch):
    from zoneinfo import ZoneInfo
    ny = ZoneInfo("America/New_York")
    def local(day, hour):
        return datetime(2026, 11, day, hour, tzinfo=ny).astimezone(UTC)
    bars = [
        (local(12, 9), 100.0),
        (local(13, 9), 103.0),
        (local(16, 9), 101.0),
        (local(17, 9), 105.0),
    ]
    monkeypatch.setattr(seq, "yahoo_bars",
                        lambda *args, **kwargs: bars)
    intraday = seq.price_return(
        object(), "SPY", local(13, 11), timedelta(hours=24), "1d",
    )
    assert intraday["baselinePrice"] == 100.0
    assert intraday["outcomePrice"] == 103.0
    after_close = seq.price_return(
        object(), "SPY", local(13, 17), timedelta(hours=24), "1d",
    )
    assert after_close["baselinePrice"] == 103.0
    assert after_close["outcomePrice"] == 101.0
    weekend = seq.price_return(
        object(), "SPY", local(14, 11), timedelta(hours=24), "1d",
    )
    assert weekend["baselinePrice"] == 103.0
    assert weekend["outcomePrice"] == 101.0


def test_older_original_publication_is_eligible_even_if_backfilled(monkeypatch, tmp_path):
    early = datetime(2026, 1, 5, tzinfo=UTC)
    rows = [sample(i, early + timedelta(days=i)) for i in range(1, 12)]
    for row in rows:
        row["firstObservedAt"] = datetime(2026, 11, 13, tzinfo=UTC).isoformat()
        row["summary"] = row["title"]
    monkeypatch.setattr(seq, "historical_news_cohort",
                        lambda **kwargs: {"items": rows})
    selected, _ = seq.load_articles(
        tmp_path / "news.sqlite3", 8,
        datetime(2026, 11, 19, tzinfo=UTC),
    )
    assert len(selected) == 8


def test_publication_time_cannot_be_after_first_observed():
    row = sample(1, datetime(2026, 11, 13, tzinfo=UTC))
    row["publishedAtReported"] = "2026-11-14T00:00:00+00:00"
    with pytest.raises(ValueError, match="Publication later"):
        seq.publication_time(row)


def test_title_only_news_does_not_generate_causal_lessons():
    class NeverCallCritic:
        def generate_json(self, **kwargs):
            raise AssertionError("Headline-only cases must not generate lessons")
    article = sample(44, datetime(2026, 11, 10, tzinfo=UTC))
    article["summary"] = article["title"]
    result = seq.next_lesson(
        NeverCallCritic(), article,
        {"impacts": [{"direction": "NEUTRAL", "priceProxy": "SPY"}]},
        {"impacts": [{"status": "WRONG", "actualDirection": "NEGATIVE"}]},
    )
    assert result["lesson"] == ""
    assert result["diagnosis"].startswith("INSUFFICIENT_ARTICLE_CONTENT")


def test_at_exact_market_close_release_is_next_day(monkeypatch):
    from zoneinfo import ZoneInfo
    ny = ZoneInfo("America/New_York")
    def nytime(day, hour):
        return datetime(2026, 11, day, hour, tzinfo=ny).astimezone(UTC)
    monkeypatch.setattr(seq, "yahoo_bars", lambda *a, **kw: [
        (nytime(12, 9), 100.0),
        (nytime(13, 9), 105.0),
        (nytime(16, 9), 110.0),
    ])
    output = seq.price_return(
        object(), "SPY", nytime(13, 16), timedelta(hours=24), "1d",
    )
    assert output["baselinePrice"] == 105.0
    assert output["outcomePrice"] == 110.0


def test_nvidia_article_rejects_unrelated_korean_ticker():
    article = sample(71, datetime(2026, 11, 11, tzinfo=UTC))
    article["title"] = "NVIDIA announces new AI accelerators"
    prediction = seq.canonical_prediction({"impacts": [
        {"targetType": "STOCK", "target": "NVDA",
         "direction": "POSITIVE", "reason": "NVIDIA product launch"},
        {"targetType": "STOCK", "target": "005930.KS",
         "direction": "POSITIVE", "reason": "Unsubstantiated spillover"},
    ]})
    result = seq.validate_asset_evidence(prediction, article)
    assert [(x["target"], x["priceProxy"]) for x in result["impacts"]] == [
        ("NVDA", "NVDA"),
    ]
    assert result["targetMappingRejections"] == [{
        "targetType": "STOCK", "target": "005930.KS",
        "reasonCode": "ARTICLE_TICKER_MISMATCH",
    }]


def test_explicitly_mentioned_us_and_korean_companies_can_coexist():
    article = sample(72, datetime(2026, 11, 11, tzinfo=UTC))
    article["title"] = "NVIDIA and SK hynix announce agreement"
    prediction = seq.canonical_prediction({"impacts": [
        {"targetType": "STOCK", "target": "NVDA",
         "direction": "POSITIVE", "reason": "Direct buyer exposure"},
        {"targetType": "STOCK", "target": "000660.KS",
         "direction": "POSITIVE", "reason": "Direct supplier exposure"},
    ]})
    assert seq.validate_asset_evidence(prediction, article)["impacts"] == prediction["impacts"]
    assert seq.asset_market("NVDA") == "US"
    assert seq.asset_market("000660.KS") == "KR"
    assert seq.asset_market("005930.KQ") == "KR"


def test_unverified_exchange_metadata_fails_closed():
    seq.verify_market_metadata("NVDA", {
        "symbol": "NVDA", "currency": "USD", "exchangeName": "NMS",
    })
    seq.verify_market_metadata("000660.KS", {
        "symbol": "000660.KS", "currency": "KRW", "exchangeName": "KSC",
    })
    with pytest.raises(ValueError, match="US_MARKET_METADATA_MISMATCH"):
        seq.verify_market_metadata("NVDA", {
            "symbol": "NVDA", "currency": "KRW", "exchangeName": "KSC",
        })
    with pytest.raises(ValueError, match="KR_MARKET_METADATA_MISMATCH"):
        seq.verify_market_metadata("000660.KS", {
            "symbol": "000660.KS", "currency": "USD", "exchangeName": "NMS",
        })
    with pytest.raises(ValueError, match="MARKET_SYMBOL_MISMATCH"):
        seq.verify_market_metadata("NVDA", {
            "symbol": "AMD", "currency": "USD", "exchangeName": "NMS",
        })


def test_ticker_suffix_without_exchange_is_not_assumed_korean():
    for ticker in ("000660", "005930", "NVDA.KS", "NVDA:US"):
        with pytest.raises(ValueError):
            seq.asset_market(ticker)
