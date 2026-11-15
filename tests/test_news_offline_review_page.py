"""Offline human reviewer is privacy-preserving and source-pinned."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from test_news_research_worker import files

from tris_market_event_intelligence.news_offline_review_page import (
    _verify_pinned_sample,
    render_offline_review,
)
from tris_market_event_intelligence.news_review_sample import prepare_news_review_sample


def _fixture(tmp_path: Path):
    cohort = files(tmp_path)[0]
    packet = prepare_news_review_sample(cohort_path=cohort, max_selected=5)
    sample = tmp_path / "review.json"
    sample.write_text(json.dumps(packet))
    return cohort, sample


def test_generates_self_contained_offline_html_without_model_calls(tmp_path: Path):
    cohort, sample = _fixture(tmp_path)
    old_bytes = sample.read_bytes()
    html = render_offline_review(cohort_path=cohort, sample_path=sample)
    assert "<!doctype html>" in html
    assert "connect-src &#x27;none&#x27;" in html
    assert "img-src &#x27;none&#x27;" in html
    assert 'id="packet-data"' in html
    assert 'id="export"' in html
    assert "JSON.stringify(output" in html
    assert 'id="import"' in html
    assert 'id="potentialMagnitude"' in html
    assert "sourceUrl" in html
    assert "http://" not in html or "http://example.org" in html
    assert "fetch(" not in html
    assert "XMLHttpRequest" not in html
    assert sample.read_bytes() == old_bytes


def test_escaping_cannot_inject_article_html_or_script(tmp_path: Path):
    cohort, sample = _fixture(tmp_path)
    original = json.loads(cohort.read_text())
    headline = '</script><img src=x onerror="alert(1)">&'
    article = original["items"][0]
    article["title"] = headline
    article["textSha256"] = hashlib.sha256(
        (headline + "\n" + (article["summary"] or "")).encode()
    ).hexdigest()
    cohort.write_text(json.dumps(original))
    sample.write_text(json.dumps(
        prepare_news_review_sample(cohort_path=cohort, max_selected=5)
    ))
    result = render_offline_review(cohort_path=cohort, sample_path=sample)
    assert headline not in result
    assert "&lt;img" not in result
    assert "\\u003c/script\\u003e" in result
    assert result.count("<script") == 2


def test_immutable_sample_metadata_change_rejected(tmp_path: Path):
    cohort, sample = _fixture(tmp_path)
    data = json.loads(sample.read_text())
    data["items"][0]["sourceUrl"] = "https://attacker.example/forged"
    sample.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="provenance"):
        render_offline_review(cohort_path=cohort, sample_path=sample)


def test_source_sha_change_rejected(tmp_path: Path):
    cohort, sample = _fixture(tmp_path)
    data = json.loads(sample.read_text())
    data["cohortSha256"] = "f" * 64
    sample.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="source cohort"):
        _verify_pinned_sample(cohort, sample)


def test_edited_sample_keeps_existing_human_fields_on_render(tmp_path: Path):
    cohort, sample = _fixture(tmp_path)
    packet = json.loads(sample.read_text())
    packet["items"][0].update({
        "importance": "IMPORTANT",
        "sourceAlignment": "SUPPORTED",
        "reviewer": "sample-editor",
        "reviewedAt": "2026-11-16T00:00:00+00:00",
        "reason": "An actual editorial human reason, not a generated label.",
    })
    sample.write_text(json.dumps(packet))
    html = render_offline_review(cohort_path=cohort, sample_path=sample)
    assert "sample-editor" in html
    assert "An actual editorial human reason" in html
