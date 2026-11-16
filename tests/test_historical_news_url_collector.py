"""Source-page fetch contract: no summaries, invented dates or price leakage."""
from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest

from tris_market_event_intelligence import historical_news_impact_test_cli as cli
from tris_market_event_intelligence import historical_news_url_collector as fetcher

URLS = [
    "https://techcrunch.com/2026/11/08/nvidia-says-its-investigating-ceo-jensen-huangs-relationship-with-possible-chinese-spy/",
    "https://www.axios.com/2026/11/08/nvidia-ceo-jensen-huang-admits-affair-board-to-launch-investigation",
]
AS_OF = "2026-11-09T03:00:00Z"


def html(*, published: str = "2026-11-08T21:00:00Z",
         modified: str = "2026-11-08T21:00:00Z") -> str:
    paragraph = (
        "Nvidia announced its own independent board investigation into "
        "media reports and allegations concerning its chief executive "
        "and a possible intelligence operation involving a business partner."
    )
    return (
        '<html><head><meta property="article:published_time" content="' + published + '"/>'
        '<meta property="article:modified_time" content="' + modified + '"/>'
        '<meta property="og:title" content="Nvidia board opens independent investigation"/>'
        '</head><body><article>'
        '<p>' + paragraph + '</p>'
        '<p>Company leadership acknowledged questions surrounding the allegations. '
        'The board said it was examining its obligations under the circumstances.</p>'
        '<p>The inquiry will review reports, seek independent advice, and '
        'determine whether further action is necessary.</p>'
        '</article></body></html>'
    )


def _mock_session(*, published: str = "2026-11-08T21:00:00Z",
                  modified: str = "2026-11-08T21:00:00Z") -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text=html(published=published, modified=modified),
        )
    return httpx.Client(transport=httpx.MockTransport(handler), trust_env=False)


def test_original_url_fetch_builds_model_input_without_hand_authored_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fetcher, "_public_dns", lambda u: None)
    with _mock_session() as session:
        packet = fetcher.collect_urls(
            URLS, as_of=AS_OF, case_id="nvda-event-2026-11-08",
            horizon=1, client=session,
        )
    assert len(packet["articles"]) == 2
    assert all("independent board investigation" in item["body"] for item in packet["articles"])
    assert all(item["publishedAt"] == datetime(2026, 11, 8, 21, tzinfo=UTC).isoformat()
               for item in packet["articles"])
    assert packet["knownTickers"] == []
    case = cli.CaseInput.model_validate(packet)
    assert cli.prepare_case(case)["articleCount"] == 2
    assert cli.prepare_case(case)["quoteAnchorCount"] >= 4


@pytest.mark.parametrize(("pub", "mod", "error"), [
    ("2026-11-09T04:00:00Z", "2026-11-09T04:00:00Z",
     "NEWS_URL_ARTICLE_AFTER_CUTOFF"),
    ("2026-11-08T21:00:00Z", "2026-11-10T10:00:00Z",
     "NEWS_URL_REVISED_AFTER_CUTOFF"),
])
def test_future_publication_and_post_cutoff_revisions_fail_closed(
    monkeypatch: pytest.MonkeyPatch, pub: str, mod: str, error: str,
) -> None:
    monkeypatch.setattr(fetcher, "_public_dns", lambda u: None)
    with (
        _mock_session(published=pub, modified=mod) as session,
        pytest.raises(ValueError, match=error),
    ):
        fetcher.collect_urls(
            URLS, as_of=AS_OF, case_id="nvda-test", horizon=1,
            client=session,
        )


def test_missing_publication_clock_is_not_fabricated() -> None:
    html_text = html().replace('property="article:published_time"', 'property="misc"')
    with pytest.raises(ValueError, match="NEWS_URL_PUBLICATION_TIME_MISSING"):
        fetcher.parse_article(html_text.encode(), URLS[0], AS_OF, "a1")


def test_paywall_or_navigation_is_not_substituted_with_summary() -> None:
    html_text = (
        '<html><head><meta property="article:published_time" '
        'content="2026-11-08T21:00:00Z"/></head>'
        '<body><h1>Subscribe to read</h1><p>Register now.</p></body></html>'
    )
    with pytest.raises(ValueError, match="NEWS_URL_ORIGINAL_BODY_NOT_AVAILABLE"):
        fetcher.parse_article(html_text.encode(), URLS[0], AS_OF, "a1")


def test_url_policy_blocks_unknown_and_local_network() -> None:
    for url in (
        "http://techcrunch.com/2026/11/08/article",
        "https://127.0.0.1/admin",
        "https://evil.example/article",
        "https://techcrunch.com:444/article",
    ):
        with pytest.raises(ValueError, match="NEWS_URL_NOT_ALLOWED"):
            fetcher._url(url)


def test_article_metadata_jsonld_supported() -> None:
    html_text = html()
    html_text = html_text.replace('property="article:published_time"', 'property="unused"')
    payload = {
        "@context": "https://schema.org", "@type": "NewsArticle",
        "datePublished": "2026-11-08T21:00:00Z",
    }
    html_text = html_text.replace(
        "</head>",
        '<script type="application/ld+json">' + json.dumps(payload) + '</script></head>',
    )
    result = fetcher.parse_article(html_text.encode(), URLS[0], AS_OF, "a1")
    assert "independent board investigation" in result["body"]

def test_rich_text_copy_identical_markdown_urls_accepted() -> None:
    original = URLS[0]
    assert fetcher._url(f"[{original}]({original})") == original
    host = "https://techcrunch.com"
    path = original.removeprefix(host)
    assert fetcher._url(f"[{host}]({host}){path}") == original


def test_rich_text_copy_mismatched_markdown_targets_rejected() -> None:
    target = URLS[0]
    with pytest.raises(ValueError, match="NEWS_URL_MARKDOWN_TARGET_MISMATCH"):
        fetcher._url(f"[https://www.axios.com/story]({target})")
    with pytest.raises(ValueError, match="NEWS_URL_MARKDOWN_TARGET_MISMATCH"):
        fetcher._url(
            "[https://www.axios.com](https://techcrunch.com)/2026/11/08/story"
        )


def test_exploratory_mode_marks_late_revised_original_not_historical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fetcher, "_public_dns", lambda u: None)
    revised = "2026-11-11T12:00:00Z"
    with _mock_session(modified=revised) as session:
        packet = fetcher.collect_urls(
            URLS, as_of=AS_OF, case_id="nvda-exploratory",
            horizon=1, client=session, allow_revised=True,
        )
    assert len(packet["articles"]) == 2
    assert all(
        item["sourceIntegrity"] == "CURRENT_PAGE_REVISED_AFTER_CUTOFF"
        for item in packet["articles"]
    )
    assert all(item["sourceModifiedAt"] == "2026-11-11T12:00:00+00:00"
               for item in packet["articles"])
    case = cli.CaseInput.model_validate(packet)
    assert case.articles[0].sourceIntegrity == "CURRENT_PAGE_REVISED_AFTER_CUTOFF"


def test_standard_mode_reports_offending_article_ordinal_and_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fetcher, "_public_dns", lambda u: None)
    with (
        _mock_session(modified="2026-11-12T12:00:00Z") as session,
        pytest.raises(
            ValueError,
            match=r"NEWS_URL_REVISED_AFTER_CUTOFF\|article=1\|host=techcrunch\.com",
        ),
    ):
        fetcher.collect_urls(
            URLS, as_of=AS_OF, case_id="nvda-strict",
            horizon=1, client=session,
        )


@pytest.mark.parametrize("container", [
    '<main><div class="wp-block-post-content">',
    '<div class="article__content">',
    '<article><div class="story-body">',
])
def test_publisher_article_main_and_wordpress_containers(
    container: str,
) -> None:
    source = html().replace("<article>", container)
    source = source.replace("</article></body>", "</div></main></body>")
    parsed = fetcher.parse_article(source.encode(), URLS[0], AS_OF, "a1")
    assert "Nvidia announced its own independent board investigation" in parsed["body"]
    assert "The inquiry will review reports" in parsed["body"]
    assert parsed["title"] == "Nvidia board opens independent investigation"


def test_publisher_main_excludes_navigation_related_links_and_marketing() -> None:
    source = html().replace(
        "<article>",
        '<main><nav><p>' + ("You may want to subscribe to every newsletter. " * 9)
        + '</p></nav><div class="article-body">',
    ).replace(
        "</article></body>",
        '</div><aside><p>' + ("Recommended subscription products here. " * 9)
        + '</p></aside></main></body>',
    )
    parsed = fetcher.parse_article(source.encode(), URLS[0], AS_OF, "a1")
    assert "You may want to subscribe" not in parsed["body"]
    assert "Recommended subscription products" not in parsed["body"]
    assert "Nvidia announced its own" in parsed["body"]


def test_nonarticle_main_marketing_not_passed_as_real_reporting() -> None:
    page = (
        '<html><head><meta property="article:published_time" '
        'content="2026-11-08T21:00:00Z"/></head><body>'
        '<main><nav><p>' + ("Subscribe for our best updates. " * 20)
        + '</p></nav><aside><p>' + ("Read more recommendations here. " * 20)
        + '</p></aside></main></body></html>'
    )
    with pytest.raises(ValueError, match="NEWS_URL_ORIGINAL_BODY_NOT_AVAILABLE"):
        fetcher.parse_article(page.encode(), URLS[0], AS_OF, "a1")
