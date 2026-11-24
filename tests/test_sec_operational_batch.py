from tris_market_event_intelligence import sec_operational_batch_cli as batch


def test_drain_stops_when_no_ready_work(monkeypatch, tmp_path):
    calls = []
    def capture(*args, **kwargs):
        calls.append("capture")
        return {"attempted": 0, "complete": 0}

    monkeypatch.setattr(batch, "collect_full_text", capture)
    monkeypatch.setattr(batch, "process", lambda *a, **k: {
        "status": "COMPLETED", "completed": 0, "failed": 0, "results": [],
    })
    monkeypatch.setattr(batch, "audit", lambda *a: {"pending": []})
    result = batch.drain(tmp_path / "news.sqlite3", "gemma4:e2b", max_batches=15)
    assert result["stopReason"] == "NO_READY_WORK"
    assert calls == ["capture"]
