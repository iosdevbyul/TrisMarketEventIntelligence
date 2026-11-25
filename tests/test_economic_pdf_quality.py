import hashlib
import sqlite3
from contextlib import closing

from tris_market_event_intelligence.economic_pdf_quality_cli import audit
from tris_market_event_intelligence.economic_pdf_recovery_cli import SCHEMA, valid_evidence

PNG = b"\x89PNG\r\n\x1a\nfixture"
TSV = "level\tpage_num\tconf\ttext\n5\t1\t45\t123\n"


def test_evidence_integrity():
    digest = hashlib.sha256(PNG).hexdigest()
    assert valid_evidence(digest, PNG, "123", TSV, "OCR_TEXT")
    assert not valid_evidence(digest, PNG + b"x", "123", TSV, "OCR_TEXT")
    assert not valid_evidence(digest, PNG, "", TSV, "OCR_TEXT")
    assert not valid_evidence(digest, PNG, "123", "", "OCR_TEXT")


def test_quality_reports_low_confidence_without_promoting(tmp_path):
    path = tmp_path / "ocr.sqlite3"
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript(SCHEMA)
        db.execute("INSERT INTO economic_pdf_recovery VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
            "source", 1, "test", 250, "kor+eng", hashlib.sha256(PNG).hexdigest(),
            PNG, "123", TSV, "OCR_TEXT", "2026-01-01",
        ))
    before = path.read_bytes()
    report = audit(path)
    assert report["valid"] == 1
    assert report["lowConfidencePages"] == 1
    assert report["numericAccuracyVerified"] is False
    assert path.read_bytes() == before
