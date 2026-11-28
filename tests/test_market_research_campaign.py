"""Campaign orchestration regression tests without local model or market data."""
import argparse
import json

import pytest

from tris_market_event_intelligence import market_research_campaign_cli as campaign
from tris_market_event_intelligence.historical_experience_abc_v2_cli import _sealed


def test_campaign_reuses_diagnostic_and_rejects_mutation(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    for name in ("autonomous-market-experience-v1", "autonomous-market-experience-v2"):
        directory = root / name
        directory.mkdir()
        _sealed(directory / "cohort.json", {"identities": []})
    _sealed(root / "autonomous-market-experience-v2" / "report.json", {"articles": 100})
    mapping = root / "semantic-market-resolution-v2"
    mapping.mkdir()
    (mapping / "expanded-adjusted-daily.csv").write_text("prices", encoding="utf-8")
    monkeypatch.setattr(campaign, "load_experiences", lambda path: [])
    calls = []

    def diagnose(development, seed, prices, output):
        calls.append("diagnose")
        output.mkdir()
        (output / "research-diagnostics.json").write_text(json.dumps({
            "prediction": {"articles": 100},
            "reflection": {"newExperiences": 158},
            "rules": {"statusCounts": {}},
        }), encoding="utf-8")

    def validate(args, client):
        assert args.exclude_run == [
            root / "autonomous-market-experience-v1",
            root / "autonomous-market-experience-v2",
        ]
        assert args.preflight is True
        return {"status": "PREFLIGHT_ONLY", "articles": 30}

    monkeypatch.setattr(campaign, "diagnose", diagnose)
    monkeypatch.setattr(campaign, "validate_run", validate)
    output = tmp_path / "output"
    args = argparse.Namespace(
        data_dir=root, output_dir=output, limit=30,
        model="gemma4:e2b", diagnostics_only=True,
    )
    assert campaign.run(args, None)["experienceCount"] == 158
    assert campaign.run(args, None)["experienceCount"] == 158
    assert calls == ["diagnose"]
    (output / "diagnostics" / "research-diagnostics.json").write_text(
        "{}", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Diagnostic report digest mismatch"):
        campaign.run(args, None)
