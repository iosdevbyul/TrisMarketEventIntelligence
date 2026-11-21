"""Fresh official BOK evidence must be reviewed before model replay."""
import hashlib

from tris_market_event_intelligence.bok_official_recollection_cli import (
    RELEASES,
    build_cohort,
    digest,
    inspect_pdf,
    quarantine_old,
)
from tris_market_event_intelligence.direct_sector_impact_cli import analyze


class LayoutPage:
    def __init__(self, body):
        self.body = body

    def extract_text(self, *, extraction_mode):
        assert extraction_mode == "layout"
        return self.body


class LayoutPDF:
    is_encrypted = False

    def __init__(self, _path, *, strict):
        assert strict
        self.pages = [
            LayoutPage("경제전망 2026\n항목 2026 Q1 2025 Q4\n" + "생산 1.2 4.2\n" * 18),
            LayoutPage("경제전망 2026\n항목 2026 Q1 2025 Q4\n" + "투자 6.6 -2.3\n" * 18),
        ]


def test_pdf_layout_keeps_page_markers_rows_and_numbers(monkeypatch, tmp_path):
    from tris_market_event_intelligence import bok_official_recollection_cli as subject

    monkeypatch.setattr(subject, "PdfReader", LayoutPDF)
    path = tmp_path / "117.pdf"
    path.write_bytes(b"%PDF-1.5\nplaceholder")
    record = RELEASES[117]
    article, audit = inspect_pdf(path, record)
    assert audit["status"] == "READY_FOR_MANUAL_TABLE_REVIEW"
    assert audit["pageCount"] == 2
    assert len(audit["pages"]) == 2
    assert "[PDF_PAGE_1]" in article["body"]
    assert "[PDF_PAGE_2]" in article["body"]
    assert "항목 2026 Q1 2025 Q4\n" in article["body"]
    assert article["officialPdfSha256"] == digest(path.read_bytes())


def test_pdf_layout_reaches_model_without_flattening():
    class Model:
        model_id = "fake"

        def generate_json(self, *, system, user, schema):
            assert "2026 Q1 2025 Q4\\n" in user
            return '{"sectors": []}'

    a = analyze({"bodyFormat": "PDF_LAYOUT", "body": "2026 Q1 2025 Q4\n투자 6.6"},
                Model())
    assert a["status"] == "ANALYZED"
    assert a["modelInputChars"] == len("2026 Q1 2025 Q4\n투자 6.6")


def test_non_pdf_is_rejected(tmp_path):
    path = tmp_path / "204.pdf"
    path.write_text("not a pdf", encoding="utf-8")
    article, audit = inspect_pdf(path, RELEASES[204])
    assert article == {}
    assert audit["status"] == "NOT_PDF"


def test_missing_cohort_never_emits_ready_for_analysis(tmp_path):
    cohort, report = build_cohort(tmp_path)
    assert cohort["articles"] == []
    assert report["status"] == "BLOCKED_INVALID_SOURCE"
    assert report["structurallyExtractedCount"] == 0
    assert len(report["items"]) == 4


def test_quarantine_moves_only_named_old_report(tmp_path):
    old = tmp_path / "old.json"
    old.write_bytes(b'{"old": true}')
    quarantine = tmp_path / "quarantine"
    result = quarantine_old(old, quarantine)
    assert result["status"] == "QUARANTINED_NOT_FOR_REUSE"
    assert not old.exists()
    assert (quarantine / old.name).read_bytes() == b'{"old": true}'
    assert result["sha256"] == hashlib.sha256(b'{"old": true}').hexdigest()
    assert quarantine_old(old, quarantine)["status"] == "OLD_REPORT_MISSING_OR_UNSAFE"


def test_no_unverified_pdf_is_silently_added(tmp_path):
    for num in (194, 204, 198):
        (tmp_path / f"{num}.pdf").write_bytes(b"not a pdf")
    cohort, report = build_cohort(tmp_path)
    assert report["status"] == "BLOCKED_INVALID_SOURCE"
    assert report["candidateCount"] == 4
    assert cohort["articles"] == []


def test_sparse_visual_page_flags_warning_without_blocking_other_pages(
    monkeypatch, tmp_path
):
    from tris_market_event_intelligence import bok_official_recollection_cli as subject

    class SparsePDF:
        is_encrypted = False

        def __init__(self, _path, *, strict):
            assert strict
            self.pages = [
                LayoutPage("경제전망 2026\\n" + "성장률 및 물가 전망.\\n" * 20),
                LayoutPage("  "),
                LayoutPage("경제전망 2026\\n" + "반도체 및 수출 업황.\\n" * 20),
            ]

    monkeypatch.setattr(subject, "PdfReader", SparsePDF)
    path = tmp_path / "117.pdf"
    path.write_bytes(b"%PDF-1.5\\nmock")
    article, audit = inspect_pdf(path, RELEASES[117])
    assert audit["status"] == "READY_FOR_MANUAL_TABLE_REVIEW"
    assert audit["emptyOrSparsePages"] == [2]
    assert audit["manualVisualReviewRequiredPages"] == [2]
    assert "SPARSE_PAGE_REQUIRES_VISUAL_REVIEW" in audit["warningCodes"]
    assert "[PDF_PAGE_2]" in article["body"]
    assert "[PDF_PAGE_3]" in article["body"]
