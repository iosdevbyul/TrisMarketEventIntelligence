from pathlib import Path

from tris_market_event_intelligence.unified_collection_launchd_cli import configurations


def test_independent_agents_and_contact_scope():
    configs = configurations(Path("/python"), Path("/repo"), Path("/news"), Path("/macro"),
                             Path("/output"), agent="Test contact test@example.org", ciks=(1,))
    assert len(configs) == 3
    for label, config in configs.items():
        job = label.rsplit(".", 1)[1]
        args = config["ProgramArguments"]
        assert args[args.index("--job") + 1] == job
        assert "--watch" in args
        assert config["KeepAlive"]
        assert ("MARKET_EVENT_SEC_USER_AGENT" in config["EnvironmentVariables"]) == (job == "sec")
