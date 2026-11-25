from tris_market_event_intelligence.global_economic_collection_cli import board_links, trusted


def test_exact_host():
    assert trusted("https://www.stats.gov.cn/english/PressRelease/", "stats.gov.cn")
    assert not trusted("https://stats.gov.cn.example.com/", "stats.gov.cn")
    assert not trusted("https://stats.gov.cn:8080/", "stats.gov.cn")


def test_board_links_excludes_external():
    raw = b'<a href="/english/PressRelease/2026/a.html">GDP</a>' \
          b'<a href="https://example.com/english/PressRelease/b.html">Other</a>'
    assert len(board_links(raw, "https://www.stats.gov.cn/english/PressRelease/",
                           "stats.gov.cn", "/english/PressRelease/")) == 1
