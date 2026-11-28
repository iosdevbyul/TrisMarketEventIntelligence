from tris_market_event_intelligence.source_aware_event_cli import extract_facts


def item(kind, title, form=None, body=""):
    return {"kind": kind, "source": "fixture", "title": title, "secForm": form,
            "documents": [{"text": body, "bodySha256": "body", "rawSha256": "raw",
                           "observedAt": "stamp"}]}


def test_form4_disposal_is_not_acquisition_or_necessarily_sale():
    body = (
        "/ownershipDocument/nonDerivativeTable/nonDerivativeTransaction/"
        "transactionAmounts/transactionShares/value: 19730\n"
        "/ownershipDocument/nonDerivativeTable/nonDerivativeTransaction/"
        "transactionAmounts/transactionAcquiredDisposedCode/value: D"
    )
    facts = extract_facts(item("SEC", "Alphabet Inc. SEC 4", "4", body))
    assert facts["documentType"] == "INSIDER_TRANSACTION"
    assert facts["transactionDirections"][0]["meaning"] == "DISPOSED"
    assert facts["transactionShareFields"][0]["value"] == "19730"
    assert not facts["predictionEligible"]


def test_prospectus_is_not_acquisition():
    assert extract_facts(item("SEC", "Alphabet SEC 424B2", "424B2"))[
        "documentType"] == "PROSPECTUS"


def test_official_release_indicator_and_entertainment_cues():
    result = extract_facts(item("ECONOMIC_RELEASE", "Services Producer Price Index (Oct.)"))
    assert result["indicators"] == [{"id": "PPI", "titleSpan": "Producer Price Index"}]
    result = extract_facts(item("NEWS", "[포토+] 배우 레드카펫"))
    assert result["marketRelevance"] == "LOW_PRIORITY_TITLE_CUE"
