"""Only verified official BOK PDF files may be downloaded."""
import httpx
import pytest

from tris_market_event_intelligence.bok_official_pdf_download_cli import (
    download_pdf,
    find_pdf_url,
    trusted,
)


def test_find_pdf_from_bok_attachment_link():
    html = '''
    <div>첨부파일
    <a href="/portal/cmmn/file/fileDown.do?id=abc">2026년 생산자물가.pdf</a>
    <a href="/portal/cmmn/file/fileDown.do?id=other">통계표.xlsx</a>
    </div>
    '''
    assert find_pdf_url(html, "https://www.bok.or.kr/portal/view.do") == (
        "https://www.bok.or.kr/portal/cmmn/file/fileDown.do?id=abc"
    )


def test_find_pdf_from_onclick_and_fail_closed_for_ambiguous_or_missing_links():
    html = """<a href="javascript:void(0)" onclick="get('/portal/data/xyz.pdf')">첨부.pdf</a>"""
    assert find_pdf_url(html, "https://www.bok.or.kr/portal/view.do") == (
        "https://www.bok.or.kr/portal/data/xyz.pdf"
    )
    with pytest.raises(ValueError, match="OFFICIAL_PDF_LINK_MISSING_OR_AMBIGUOUS"):
        find_pdf_url('<a href="#">첨부.pdf</a>', "https://www.bok.or.kr")
    with pytest.raises(ValueError, match="OFFICIAL_PDF_LINK_MISSING_OR_AMBIGUOUS"):
        find_pdf_url('<a href="https://evil.test/abc.pdf">첨부.pdf</a>', "https://www.bok.or.kr")


def test_only_trusted_https_hosts():
    assert trusted("https://www.bok.or.kr/portal/file.pdf")
    assert trusted("https://file-cdn.bok.or.kr/portal/abc.pdf")
    assert not trusted("http://www.bok.or.kr/portal/file.pdf")
    assert not trusted("https://bok.or.kr.evil.test/file.pdf")


def test_download_verifies_pdf_signature_and_preserves_existing_files(tmp_path):
    def respond(request):
        if request.url.path.endswith("view.do"):
            return httpx.Response(200, text=(
                '<a href="/portal/files/abc.pdf">[보도자료] 2026년 5월.pdf</a>'
            ))
        return httpx.Response(200, content=b"%PDF-1.4\n" + b"test-data")

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        target = tmp_path / "194.pdf"
        result = download_pdf(
            client, "https://www.bok.or.kr/portal/bbs/view.do", target
        )
        assert result["status"] == "DOWNLOADED"
        assert target.read_bytes().startswith(b"%PDF-")
        assert download_pdf(
            client, "https://www.bok.or.kr/portal/bbs/view.do", target
        )["status"] == "EXISTING_FILE_NOT_OVERWRITTEN"


def test_html_response_must_not_be_saved_as_pdf(tmp_path):
    def respond(request):
        if request.url.path.endswith("view.do"):
            return httpx.Response(200, text='<a href="/file.pdf">자료.pdf</a>')
        return httpx.Response(200, text="<html>blocked</html>")

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        target = tmp_path / "fake.pdf"
        with pytest.raises(ValueError, match="INVALID_OR_OVERSIZED_PDF_RESPONSE"):
            download_pdf(client, "https://www.bok.or.kr/view.do", target)
        assert not target.exists()
