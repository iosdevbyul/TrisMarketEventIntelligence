# Phase 5-D5 — Independent collected-news quality audit

The 10 exact-headline replay examples only show that previously known titles
match lexical patterns; they do **not** demonstrate recall on unseen articles.
The live probe has collected 111 RSS articles but cannot establish whether
the unmatched items contain missed events without human review.

A local command now samples **100 real collected articles**, stratified by
source with a reproducible seed:

```bash
market-event-quality --db data/news.sqlite3 --export private/review.csv --count 100
```

The export contains article URLs, titles and source capture hashes to enable
manual ground-truth review. Keep it **local/private**; never commit or upload
it to public GitHub Actions. Fill the `gold_categories` column for *every*
row with `NONE` (no detected market-event claim) or one or more of the
existing event category names, separated with `|` (e.g. `EARNINGS|REGULATION`).
**Do not copy the predicted column into gold without independent verification.**

```bash
market-event-quality --db data/news.sqlite3 --score private/review.csv
```

Scoring checks the articles remain unchanged, rejects incomplete labels,
recomputes predictions and outputs per-source false positives, false
negatives, micro precision and recall. An empty positive sample returns
null recall (not a fraudulent 100%). It does not expose article titles or
raw content in the numeric scoring report.

Acceptance still requires independent labeling of at least 100 articles,
a positive reference sample, agreement on quality thresholds, and sustained
unattended source discovery. Do not equate a diagnostic pass with
`gate_6_ready`; that is always false in this tool.

BLS RSS returns HTTP 403 in the GitHub runner; no access bypass is attempted.
The provider publishes official PPI information at
https://www.bls.gov/ppi/ and RSS listings at https://www.bls.gov/feed/.
The official web presence does not establish HTML-collector permission.
GDELT DOC has responded with rate limits or timeouts; the existing collector
stops further searches on HTTP 429 and does not evade provider controls.

No AI inference, no macOS background service, no Donghak modifications.
