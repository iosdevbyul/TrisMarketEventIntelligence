"""Regression tests for retrospective BOK attachment evidence capture."""
import hashlib

import httpx
import pytest

from tris_market_event_intelligence.bok_attachment_evidence_cli import (
    acquire, allowed, attachment_links, capture,
)


def article():
    return {
        "articleId": 42, "source": "bok_statistics",
        "sourceUrl": "https://www.bok.or.kr/board/press/view.do?id=42",
        "publishedAtReported": "2026-05-10T09:00:00+09:00",
    }


def client_for(download=None):
    text = (download if download is not None else
            b"Official statistical attachment: quarterly growth amounted "
            b"to 2.5 percent, compared with the preceding quarter.")
    def handler(req):
        if req.url.path.endswith("view.do"):
            return httpx.Response(
                200, headers={"content-type": "text/html"},
                text='<a href="/file/download.do?id=100">첨부자료 다운로드</a>'
                     '<a href="javascript:download(10)">붙임</a>'
                     '<a href="https://other.example/b.pdf">첨부</a>',
            )
        return httpx.Response(200, headers={"content-type": "text/plain"}, content=text)
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_unsafe_or_script_links_never_downloaded():
    assert not allowed("https://www.bok.or.kr.evil.example/file.pdf")
    assert not allowed("http://www.bok.or.kr/file.pdf")
    assert not allowed("https://user@www.bok.or.kr/file.pdf")
    found, errors = attachment_links(
        '<a href="javascript:download(2)">첨부</a>'
        '<a href="https://evil.example/file.pdf">첨부</a>',
        article()["sourceUrl"],
    )
    assert not found
    assert {x["reason"] for x in errors} == {"SCRIPT_ONLY_LINK", "UNSAFE_LINK"}


def test_real_bytes_and_extracted_text_are_hashed_without_pit_claim():
    with client_for() as client:
        result = capture(article(), client)
    assert result["status"] == "TEXT_EVIDENCE_ACQUIRED"
    assert not result["firstPublicationPITCertified"]
    attachment = result["attachments"][0]
    assert attachment["extractionStatus"] == "TEXT_EXTRACTED"
    assert attachment["rawSha256"] == hashlib.sha256(
        attachment["extractedText"].encode(),
    ).hexdigest()
    assert len(result["unresolvedLinks"]) == 2


def test_unsupported_or_insufficient_text_not_accepted():
    with client_for(b"short") as client:
        result = capture(article(), client)
    assert result["status"] == "ATTACHMENTS_UNVERIFIED"
    assert result["attachments"][0]["extractionStatus"] == "NO_SUBSTANTIVE_TEXT"


def test_redirect_is_not_followed():
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(302, headers={"location": "https://evil.example"})
    )) as client:
        result = capture(article(), client)
    assert result["status"] == "PAGE_RETRIEVAL_FAILED"
    assert result["errorCode"] == "UNSAFE_BOK_REDIRECT_DESTINATION"


def test_only_review_needed_frozen_articles_are_requested():
    first = article()
    second = {**article(), "articleId": 43}
    report = {
        "articleCount": 2, "originalHoldoutSha256": "sealed",
        "articles": [
            {"articleId": 42, "sourceReadiness": "SOURCE_REVIEW_REQUIRED"},
            {"articleId": 43, "sourceReadiness": "SOURCE_READY"},
        ],
    }
    with client_for() as client:
        output = acquire([first, second], report, client)
    assert output["targetedCount"] == 1
    assert output["pointInTimeCertified"] is False
    assert output["originalPredictionsModified"] is False
    with client_for() as client, pytest.raises(ValueError, match="COHORT_IDENTITY_MISMATCH"):
        acquire([first, second], {**report, "articles": report["articles"][:1]}, client)


def test_bok_cdn_redirect_stays_inside_approved_hosts():
    def handler(request):
        if request.url.host == "www.bok.or.kr":
            return httpx.Response(302, headers={
                "location": "https://file-cdn.bok.or.kr/portal/a/1/report.txt?token=temporary"
            })
        return httpx.Response(200, headers={"content-type": "text/plain"},
                              content=b"Verified source content " * 8)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        from tris_market_event_intelligence.bok_attachment_evidence_cli import fetch
        content, mime = fetch(
            client, "https://www.bok.or.kr/file/download.do?id=42",
            2000, file_url=True,
        )
    assert len(content) > 50
    assert mime == "text/plain"


def test_download_failure_is_specific_http_status():
    def handler(request):
        if request.url.path.endswith("view.do"):
            return httpx.Response(
                200, headers={"content-type": "text/html"},
                text='<a href="/file/download.do?id=42">첨부자료 다운로드</a>',
            )
        return httpx.Response(403, text="access denied")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = capture(article(), client)
    assert result["attachments"][0]["errorCode"] == "HTTP_403"
    assert result["status"] == "ATTACHMENTS_UNVERIFIED"


def test_external_redirect_is_still_rejected_after_fix():
    def handler(request):
        if request.url.path.endswith("view.do"):
            return httpx.Response(
                200, headers={"content-type": "text/html"},
                text='<a href="/file/download.do?id=42">첨부자료 다운로드</a>',
            )
        return httpx.Response(302, headers={"location": "https://untrusted.example/evil.pdf"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = capture(article(), client)
    assert result["attachments"][0]["errorCode"] == "UNSAFE_BOK_REDIRECT_DESTINATION"


def test_pdf_uses_existing_pypdf_dependency_without_fitz():
    import io
    from pypdf import PdfWriter
    from tris_market_event_intelligence.bok_attachment_evidence_cli import (
        attachment_format, extract,
    )

    output = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.write(output)
    raw = output.getvalue()
    assert attachment_format(raw, "application/octet-stream") == "PDF"
    text, status = extract(raw, "application/octet-stream", "https://www.bok.or.kr/file/download.do")
    assert not text
    assert status == "NO_SUBSTANTIVE_TEXT"


def test_hwpx_sections_extract_actual_xml_text():
    import io
    import zipfile
    from tris_market_event_intelligence.bok_attachment_evidence_cli import (
        attachment_format, extract,
    )

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("mimetype", "application/hwp+zip")
        archive.writestr(
            "Contents/section0.xml",
            '<hp:sec xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph">'
            '<hp:t>한국은행 통계 발표에서 국내총생산이 전기 대비 2.5% 증가했다는 내용입니다.</hp:t>'
            '<hp:t> 수치의 발표 시기와 비교 기준을 따로 확인해야 합니다.</hp:t>'
            '</hp:sec>',
        )
    raw = payload.getvalue()
    assert attachment_format(raw, "application/octet-stream") == "HWPX"
    text, status = extract(raw, "application/octet-stream", "https://www.bok.or.kr/file/download.do")
    assert status == "TEXT_EXTRACTED"
    assert "2.5%" in text


def test_legacy_hwp_and_unknown_zip_remain_unverified():
    import io
    import zipfile
    from tris_market_event_intelligence.bok_attachment_evidence_cli import extract

    text, status = extract(
        bytes.fromhex("d0cf11e0a1b11ae1") + b"unverified",
        "application/octet-stream",
        "https://www.bok.or.kr/file/download.do",
    )
    assert text == ""
    assert status == "HWP_OLE_EXTRACTOR_UNAVAILABLE"
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("file.txt", "not a verified document format")
    text, status = extract(
        payload.getvalue(), "application/octet-stream",
        "https://www.bok.or.kr/file/download.do",
    )
    assert text == ""
    assert status == "UNSUPPORTED_ZIP_UNKNOWN_FORMAT"
