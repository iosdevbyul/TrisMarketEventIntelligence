import pytest

from tris_market_event_intelligence.ai_semantic_resolution_cli import (
    candidates,
    score_choices,
    validate_choices,
)


def test_multi_target_candidates_and_source_reference():
    master = {"instruments": [
        {"name": "삼성전자", "market": "XKRX", "symbol": "005930.KS",
         "benchmark": "^KS11", "source": "https://kind.krx.co.kr/"},
        {"name": "SK하이닉스", "market": "XKRX", "symbol": "000660.KS",
         "benchmark": "^KS11", "source": "https://kind.krx.co.kr/"},
    ]}
    options = candidates("Samsung Electronics, SK Hynix", "", "", master)
    assert {x["symbol"] for x in options} == {"005930.KS", "000660.KS"}
    passages = [{"id": "E1", "text": "Both companies face higher costs."}]
    choices = [{"candidate_id": x["id"], "evidence_id": "E1", "reason": "Both named"}
               for x in options]
    assert len(validate_choices({"choices": choices}, options, passages)) == 2
    choices[0]["candidate_id"] = "C999"
    with pytest.raises(ValueError, match="Invalid or duplicate"):
        validate_choices({"choices": choices}, options, passages)


def test_score_preserves_direction(tmp_path):
    prices = tmp_path / "prices.csv"
    prices.write_text(
        "symbol,date,adjusted_close,source\n"
        "AAA,2026-01-05,100,test\nAAA,2026-01-06,90,test\n"
        "BBB,2026-01-05,100,test\nBBB,2026-01-06,110,test\n"
        "SPY,2026-01-05,100,test\nSPY,2026-01-06,100,test\n",
        encoding="utf-8",
    )
    rows = score_choices([{
        "id": "example", "articleId": 1, "impactIndex": 0, "target": "A and B",
        "direction": "DOWN", "publishedAt": "2026-01-04T12:00:00+00:00",
        "choices": [{"symbol": symbol, "market": "XNAS", "benchmark": "SPY",
                     "mappingKind": "DIRECT_EQUITY"} for symbol in ("AAA", "BBB")],
    }], prices)
    scored = [row for row in rows if row["sessions"] == 1]
    assert [row["verdict"] for row in scored] == ["CORRECT", "INCORRECT"]
    assert all(row["direction"] == "DOWN" for row in rows)
