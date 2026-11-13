# Phase 5-D1/D2 — Discovery and conservative article grouping

The existing RSS polling engine is unchanged. A new optional discovery command
adds Korean news search (Naver Search API) and global indexed news (GDELT DOC 2.0).
This is **discovery**, not primary-source authentication or event extraction.

```bash
# Without NAVER_CLIENT_ID and NAVER_CLIENT_SECRET, Naver reports not_configured.
market-event-discover --include-rss
market-event-discover --skip-naver
market-event-discover --index-only
```

Naver credentials belong in private local environment variables, never Git.
Only date-sorted top 100 results per configured search are requested. GDELT
uses ArticleList JSON with a rolling 1-day time window and 75 results per query.
Search topics cover macro, companies, sectors and geopolitical disruptions.
Both sources are optional; an unavailable source must not block other sources.
Network payloads and first-detection timestamps are preserved in existing
captures/articles tables. Publisher permissions, quota and attribution rules
still apply; no paid article content is downloaded.

Indexing does not overwrite source articles or assign event verification.
The new story_groups and story_members tables preserve every original article.
Exact canonical URL matches, or exact *distinctive* five-token-plus headlines on
the same UTC publication day, are grouped as **story candidates**, not confirmed
identical events. Weak titles and undated articles remain separate. Corrections
and reports with different wording may remain separate deliberately.
No external text embedding or paid LLM is used.

Operational limitations:
- Naver Search requires developer API credentials and publisher terms review.
- GDELT coverage is selective, time-windowed and cannot guarantee an exhaustive
  historical reconstruction or that all desired 10 news stories are included.
- Query ranking and top-N pagination can omit stories; gap/coverage measurement is
  required before enabling unattended production polling.
- Story group IDs are database-scoped and not equivalent to verified event IDs.
- Consumer API still exposes articles' extracted events, not story groups.
- Existing saved articles can be grouped with --index-only.
- Live provider success, reliability and replay on verified original news remain
  **unproven** by offline CI. AI impact analysis is not invoked here.
