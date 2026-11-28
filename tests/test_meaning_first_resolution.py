import pytest

from tris_market_event_intelligence.meaning_first_resolution import (
    match_entities,
    validate_classification,
)

MASTER = {"instruments": [
    {"name": "Gartner, Inc.", "market": "XNYS", "symbol": "IT",
     "benchmark": "SPY", "source": "directory"},
    {"name": "Financial Institutions, Inc.", "market": "XNAS", "symbol": "FISI",
     "benchmark": "SPY", "source": "directory"},
    {"name": "The Bancorp, Inc.", "market": "XNAS", "symbol": "TBBK",
     "benchmark": "SPY", "source": "directory"},
    {"name": "삼성전자", "market": "XKRX", "symbol": "005930.KS",
     "benchmark": "^KS11", "source": "directory"},
    {"name": "SK하이닉스", "market": "XKRX", "symbol": "000660.KS",
     "benchmark": "^KS11", "source": "directory"},
]}


def packet(target, quote):
    return {"target": target, "sourcePassages": [{"id": "E1", "text": quote}]}


def answer(kind, name, canonical):
    return {"kind": kind, "reason": "Source identification", "entities": [
        {"name": name, "canonical_name": canonical, "evidence_id": "E1"},
    ]}


@pytest.mark.parametrize(("target", "name", "canonical"), [
    ("IT Sector", "IT", "Gartner"),
    ("Financial Institutions", "Financial Institutions", "Financial Institutions"),
    ("Mutual Federal Bancorp", "Bancorp", "The Bancorp"),
    ("Consumer Goods sector (e.g., vehicles, home appliances)", "e.g.", "Genpact"),
])
def test_no_unrelated_company(target, name, canonical):
    assert match_entities(
        answer("EQUITY", name, canonical),
        packet(target, "The economic statistics describe " + target), MASTER,
    ) == []


def test_no_bond_to_stock_index():
    assert match_entities(
        answer("INDEX", "Bond Market", "KOSPI"),
        packet("Bond Market", "Bond Market issuance increases this month"), MASTER,
    ) == []


def test_page_marker_rejected():
    with pytest.raises(ValueError, match="Non-substantive"):
        validate_classification(
            answer("EQUITY", "삼성전자", "삼성전자"),
            packet("삼성전자", "[PDF PAGE 1]"),
        )


def test_two_named_companies():
    result = answer("EQUITY", "삼성전자", "삼성전자")
    result["entities"].append({
        "name": "SK하이닉스", "canonical_name": "SK하이닉스", "evidence_id": "E1",
    })
    choices = match_entities(
        result, packet("삼성전자, SK하이닉스", "삼성전자와 SK하이닉스의 비용이 상승했다."), MASTER,
    )
    assert {x["symbol"] for x in choices} == {"005930.KS", "000660.KS"}
