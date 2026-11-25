from tris_market_event_intelligence.global_economic_collection_cli import board_links


def test_customs_requires_real_detail_identifier():
    base = "https://customs.go.kr/"
    path = "/kcs/na/ntt/selectNttInfo.do"
    html = (
        f'<a href="{path}?nttSn=123&amp;nttSnUrl=abc">Trade release</a>'
        f'<a href="{path}?nttSn=124">Missing token</a>'
    )
    assert board_links(html.encode(), base, "customs.go.kr", path) == {
        "https://customs.go.kr/kcs/na/ntt/selectNttInfo.do?nttSn=123&nttSnUrl=abc": "Trade release",
    }
