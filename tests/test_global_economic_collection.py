from tris_market_event_intelligence.global_economic_collection_cli import (
    board_links,
    feed_records,
    trusted,
)


def test_exact_host():
    assert trusted("https://www.stats.gov.cn/english/PressRelease/", "stats.gov.cn")
    assert not trusted("https://stats.gov.cn.example.com/", "stats.gov.cn")
    assert not trusted("https://stats.gov.cn:8080/", "stats.gov.cn")


def test_board_links_excludes_external():
    raw = b'<a href="/english/PressRelease/2026/a.html">GDP</a>' \
          b'<a href="https://example.com/english/PressRelease/b.html">Other</a>'
    assert len(board_links(raw, "https://www.stats.gov.cn/english/PressRelease/",
                           "stats.gov.cn", "/english/PressRelease/")) == 1


def test_rdf_feed_https_normalization():
    raw = b"""<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
      xmlns="http://purl.org/rss/1.0/"><item><title>Policy</title>
      <link>http://www.boj.or.jp/en/policy.htm</link></item></rdf:RDF>"""
    records = feed_records("boj", raw, "https://www.boj.or.jp/en/rss/whatsnew.xml",
                           "boj.or.jp", "2026-01-01")
    assert records[0].url == "https://www.boj.or.jp/en/policy.htm"
    assert records[0].published_at is None
