# Historical GDELT GKG source scout — bounded opt-in exploration

This is the first actual 10-year **historical NEWS reference discovery** capability
following the operator's 2016–2026 real NVDA and SPY Tiingo acquisition
and successful 100 shock / 100 quiet-day study.

Reported baseline from real local execution on 2026-11-15:

- 2513 common price sessions and 2512 candidate daily moves
- 100 highest absolute NVDA–SPY excess-return shock dates
- TOP 1 = 2023-05-25, NVDA +24.3696%, SPY +0.8660%
- 100 control dates
- **zero** matching historical pre-open NEWS; economic indicators not provided
- local research packet at
  `$HOME/Library/Application Support/TrisMarketEventIntelligence/nvda-spy-shocks-2016-2026.json`

## Source and restrictions

GDELT 2.0 GKG 2.1 archive documentation:
https://gdeltproject.org/data.html and
https://data.gdeltproject.org/documentation/GDELT-Global_Knowledge_Graph_Codebook-V2.1.pdf .

GDELT archives 15-minute metadata batches dating to 2015. This scout
selects **four** isolated archive slots relative to the chosen selected
NYSE opening (1 day, 12 hours, 4 hours and 1 hour before). It downloads
only those selected GKG ZIPs when explicitly run, not a full decade,
and never scrapes publisher article text or invokes an LLM.

Only WEB source rows with a GKG **organization** mention matching
'NVIDIA' are kept. Output preserves URL, source name, GKG publication
date field and batch time plus ZIP SHA256. Data are **candidates**, not
verified corporate issuer links or independently checked articles.
The GKG batch slot must not be treated as proof the user could have
accessed an article at that exact instant; archive availability may lag.
No source article copyright permission, completeness or causal identity
is inferred. Include citation to The GDELT Project for any downstream use.

## Private opt-in Mac example

Preview first without contacting GDELT:

```sh
bash scripts/news-learning-review.sh --update gdelt-scout \
  --study "$HOME/Library/Application Support/TrisMarketEventIntelligence/nvda-spy-shocks-2016-2026.json" \
  --rank 1 --preview
```

Then explicitly retrieve the four GKG metadata ZIPs (network usage):

```sh
bash scripts/news-learning-review.sh gdelt-scout \
  --study "$HOME/Library/Application Support/TrisMarketEventIntelligence/nvda-spy-shocks-2016-2026.json" \
  --rank 1 --run \
  --output "$HOME/Library/Application Support/TrisMarketEventIntelligence/nvda-gdelt-gkg-rank1.json"
```

The scout writes a new create-only mode-0600 JSON artifact, not a database.
Do not rerun with the same output path. In case of historical unavailable
batches or provider errors the command fails instead of manufacturing
articles. Do not claim causal attribution or use URL mentions as Gold.

## Next work

Check actual returned archive ZIPs, understand whether metadata is
sufficient and revise the sampling window based on actual coverage.
GKG contains **references**, not full articles; source-text licensing
requires separate review. Expand to selected shock days and matched
controls with conservative download budgets, deduplication, contextual
macro economic event source archives, and explicit source availability
chronology. Only then provide **historically available** evidence to
Gemma's retrospective hypothesis workflow and independently test
sealed forward predictions. 24/7, trading, training and promotion
remain disabled until independently verified and authorized.
