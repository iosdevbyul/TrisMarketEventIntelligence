import json

import pytest

from tris_market_event_intelligence.event_profile_integration_cli import (
    ValidationError,
    profile_pairs,
    validate_profile,
    validate_review,
)


def test_profile_matching_rejects_incidental_company_and_wrong_period():
    base = {"publisherDeclaredAt": "2026-11-20T12:00:00+00:00", "secAcceptanceAt": None,
            "canonicalUrl": None}
    a = {**base, "eventId": "a", "source": "sec", "kind": "SEC"}
    b = {**base, "eventId": "b", "source": "news", "kind": "NEWS"}
    p = {"actor": "NVIDIA", "event_type": "EARNINGS", "period": "2026-Q3",
         "action": "revenue growth", "object": "quarterly results", "key_fact": "revenue 50 billion"}
    assert len(list(profile_pairs([(a, "p1", p), (b, "p2", {**p, "actor": "엔비디아"})]))) == 1
    assert not list(profile_pairs([(a, "p1", p), (b, "p2", {**p, "actor": "SpaceX"})]))
    assert not list(profile_pairs([(a, "p1", p), (b, "p2", {**p, "period": "2026-Q2"})]))


def test_source_ids_instead_of_generated_quotes():
    passage = {"id": "1", "text": "NVIDIA quarterly revenue grew."}
    p = {"actor": "NVIDIA", "event_type": "EARNINGS", "action": "grew", "object": "revenue",
         "period": "", "key_fact": "revenue grew", "evidence_id": "1"}
    assert validate_profile(json.dumps(p), [passage])["evidence"] == passage
    with pytest.raises(ValidationError, match="INVALID_EVIDENCE_ID"):
        validate_profile(json.dumps({**p, "evidence_id": "99"}), [passage])
    r = {"relation": "SAME_EVENT", "left_id": "1", "right_id": "1",
         "reason": "same reported revenue", "new_fact": ""}
    assert validate_review(json.dumps(r), [passage], [passage])["leftEvidence"] == passage
