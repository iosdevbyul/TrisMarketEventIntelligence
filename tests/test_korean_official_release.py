from tris_market_event_intelligence.korean_official_release_cli import attachments, discover


def test_only_official_release_links():
    raw = b'<a href="/board.es?act=view&amp;list_no=123&amp;bid=216">Release</a>' \
          b'<a href="https://example.com/board.es?act=view&amp;list_no=1">Bad</a>'
    assert discover(raw, "https://mods.go.kr/board.es") == [
        ("https://mods.go.kr/board.es?act=view&list_no=123&bid=216", "Release")
    ]


def test_attachment_links_keep_original_query():
    raw = b'<a href="/boardDownload.es?bid=216&amp;list_no=123&amp;seq=1">PDF</a>'
    assert attachments(raw, "https://mods.go.kr/board.es") == [
        "https://mods.go.kr/boardDownload.es?bid=216&list_no=123&seq=1"
    ]
