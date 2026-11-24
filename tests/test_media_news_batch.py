"""Media-only isolation and explicit coverage of blocked source evidence."""
import hashlib
import sqlite3

from tris_market_event_intelligence.automatic_news_pipeline import initialize, pending
from tris_market_event_intelligence.media_news_batch_cli import audit


def test_media_scope_and_audit(tmp_path):
    path = tmp_path / "news.sqlite3"
    body = "The company cut its profit forecast."
    raw = body.encode()
    digest = hashlib.sha256(raw).hexdigest()
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE articles (id INTEGER PRIMARY KEY,source TEXT,title TEXT,
                url TEXT,published_at TEXT,detected_at TEXT);
            CREATE TABLE article_full_text (article_id INTEGER,status TEXT,body TEXT,
                body_sha256 TEXT,raw_payload BLOB,raw_sha256 TEXT,sparse_pages TEXT,error_code TEXT);
            INSERT INTO articles VALUES
                (1,'bbc_business','Forecast','https://www.bbc.com/news/a',NULL,NULL),
                (2,'fed_all','Rate','https://www.federalreserve.gov/a',NULL,NULL),
                (3,'sec_filing_789019','Filing','https://www.sec.gov/a',NULL,NULL),
                (4,'bbc_world','Video','https://www.bbc.com/news/videos/a',NULL,NULL);
        """)
        initialize(db)
        for aid in (1, 2, 3):
            db.execute("INSERT INTO article_full_text VALUES (?,'COMPLETE',?,?,?,?, '[]',NULL)",
                       (aid, body, digest, raw, digest))
        db.execute("INSERT INTO article_full_text VALUES (4,'NON_ARTICLE',NULL,NULL,NULL,NULL,'[]',?)",
                   ("NON_ARTICLE_MEDIA_PAGE",))
        rows, _ = pending(db, "gemma4:e2b", 12, 30, media_only=True)
        assert [row["id"] for row in rows] == [1]
    report = audit(path, "gemma4:e2b")
    assert report["discovered"] == 2
    assert report["counts"] == {"READY_FOR_ANALYSIS": 1, "EXCLUDED_AUDIO_VIDEO": 1}
