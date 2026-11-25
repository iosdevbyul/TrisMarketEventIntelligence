from types import SimpleNamespace

from tris_market_event_intelligence import unified_collection_cli as supervisor


def test_sec_missing_contact_blocks_without_network(monkeypatch, tmp_path):
    monkeypatch.delenv("MARKET_EVENT_SEC_USER_AGENT", raising=False)
    assert supervisor.sec_job(tmp_path / "news.sqlite3", (1,), 7, 1)["status"] == "BLOCKED"


def test_failed_job_records_status_and_other_job_can_run(monkeypatch, tmp_path):
    def fail(*args):
        raise ValueError("private data")

    monkeypatch.setattr(supervisor, "news_job", fail)
    monkeypatch.setattr(supervisor, "economic_job", lambda args: {"status": "OK"})
    args = SimpleNamespace(news_db=tmp_path / "news.sqlite3", batches=1, output_dir=tmp_path)
    failed = supervisor.run_job(args, "news")
    assert failed["status"] == "FAILED"
    assert failed["error"] == "ValueError"
    assert failed["modelCalls"] == 0
    assert supervisor.run_job(args, "economic")["status"] == "OK"
    assert (tmp_path / "news-latest.json").is_file()
