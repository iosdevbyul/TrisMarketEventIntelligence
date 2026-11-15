# Learning-05: Offline browser review of the 48 real NEWS candidates

The source-pinned 48-article local packet exists at
`~/Library/Application Support/TrisMarketEventIntelligence/news-review-sample-20261115.json`.
The 482-row blank importance and potential magnitude templates already exist.
Do NOT recreate data or invent human gold.

This command renders an HTML document stored only on the user's Mac.
No HTTP server, AI model, extension, external JS/CSS, analytics, CDN,
remote font or login is required. The browser page has an explicit Content
Security Policy that denies network connection, remote assets and forms.
The **original news link can be opened by an intentional click** in a new
tab, which may contact the article's source website.

```sh
bash scripts/news-learning-review.sh --update review-page \
  --cohort "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-cohort-20261115.json" \
  --sample "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-review-sample-20261115.json" \
  --output "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-review-page-20261115.html"
```

Open that local HTML in your web browser using Finder (or explicitly
`open "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-review-page-20261115.html"`).
Review each news title, summary and original source before assessing it.
Fill in importance/source alignment, potential magnitude, whether any
same-story grouping is independently supported, reviewer identifier
and a substantive 20+ character reason.

The page does not autosave and no data is transmitted. **Export**
`news-review-human-edited.json` explicitly after a session. The file is
saved by your browser (typically Downloads); put that file in a chosen
private folder on your Mac and point the existing bridge at its exact
path. To resume, reopen the private HTML, then **import** the previously
exported JSON (it must match all original immutable fields and pair order).
Always export again after new edits. Do not upload the HTML or JSON to
public repos, Trello or a website.

The UI blocks partial importance/source-alignment edits, missing
reviewer/timestamp/reason, invalid magnitude and altered source data
when reimporting. The existing Python review bridge is the **authoritative**
integrity check and rejects unsupported values, forged source fields,
source hash changes and missing human evidence. It will NEVER fabricate
human evaluation or automatically duplicate one label across multiple
articles.

Before exporting to full 482-row review files, run (substitute a real
private edited JSON path):

```sh
bash scripts/news-learning-review.sh review-bridge --audit \
  --cohort "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-cohort-20261115.json" \
  --sample "$HOME/Downloads/news-review-human-edited.json" \
  --importance-template "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-human-review-20261115.json" \
  --materiality-template "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-materiality-review-20261115.json"
```

If audit succeeds and real human judgments exist, use the documented
`review-bridge --apply` with explicit *new* private importance and
magnitude output files; it refuses blank exports and overwriting old
outputs. A human judgment is still self-attested, not audited gold.
No stock prices, OHLCV, volumes, returns, future news labels,
production DB changes, automatic training or model promotion.
