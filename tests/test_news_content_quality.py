import hashlib

import pytest

from tris_market_event_intelligence.news_content_quality import classify


@pytest.mark.parametrize("body", [
    "긴급속보: 젠슨 황 CEO 복귀",
    "Fed cuts rates 50 bps.",
    "금리 동결",
])
def test_short_factual_text_is_not_rejected_by_length(body):
    assert classify(body) == "TEXT_CANDIDATE"


@pytest.mark.parametrize("body", [
    "", "   ", "Only a title", "Please sign in to continue",
    "Watch the video", "광고", "자세히 보기",
])
def test_content_without_information_is_not_admitted(body):
    assert classify(body) != "TEXT_CANDIDATE"


def test_media_shell_is_not_a_verified_transcript():
    assert classify(
        "How Nvidia is reacting to its CEO's return",
        url="https://www.bbc.co.uk/news/videos/example",
    ) == "MEDIA_REVIEW_REQUIRED"


def test_digest_and_partial_content_are_checked():
    body = "Fed cuts rates."
    digest = hashlib.sha256(body.encode()).hexdigest()
    assert classify(body, digest=digest) == "TEXT_CANDIDATE"
    assert classify(body, digest="bad") == "DIGEST_MISMATCH"
    assert classify(body, sparse="[1]") == "PARTIAL_CONTENT"


@pytest.mark.parametrize("path", ["/sounds/play/episode", "/iplayer/live/bbcnews",
                                  "/news/videos/clip", "/news/av/clip"])
def test_audio_video_excluded_from_text_admission(path):
    assert classify("Publisher programme description.", url="https://www.bbc.co.uk" + path) != "TEXT_CANDIDATE"
