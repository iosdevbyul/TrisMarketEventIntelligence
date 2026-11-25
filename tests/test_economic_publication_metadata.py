from tris_market_event_intelligence.global_economic_collection_cli import PublicationMetadata


def test_publication_metadata_keeps_declared_date_and_revision_separate():
    parser = PublicationMetadata()
    parser.feed(
        '<meta property="article:published_time" content="2026-01-02T09:00:00+09:00">'
        '<meta property="article:modified_time" content="2026-01-03T09:00:00+09:00">'
        '<time datetime="2026-01-02"></time>'
    )
    assert parser.values == [
        ("article:published_time", "2026-01-02T09:00:00+09:00"),
        ("article:modified_time", "2026-01-03T09:00:00+09:00"),
        ("time", "2026-01-02"),
    ]
