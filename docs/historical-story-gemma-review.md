# Historical NEWS story candidates and optional local Gemma review

## Grounded first case (real operator research, 2026-11-15)

The user supplied real `nvda-gdelt-gkg-rank1.json` after collecting
NVDA/SPY adjusted prices and selecting 2023-05-25 as their highest
absolute daily benchmark-excess move (+23.5036 percentage points).
The GDELT archive metadata contains **44 distinct URLs across four 15-minute
archive slots**, with per-slot matching reference counts 14, 10, 9, 11.
There is no full article text, source-published clock, proved corporate link,
price attribution or authorized economic indicator revision evidence.

The data show specific candidate URL paths:

- Yahoo: `nvidia-sees-two-simultaneous-transitions`
- RTE: `nvidia-quarterly-results`
- Wccftech: `shares-surge-by-27...`
- StreetInsider/Reuters: `Nasdaq futures jump as Nvidia leads AI-driven rally on upbeat forecast`
- MSN: `nvidia-s-blowout-forecast-sparks-260-billion-ai-rally`
- Several regional articles reprint "rock-and-roll icon Tina Turner dies, Nvidia
  shares spike, Real Madrid..." (possible same syndicated story; not verified)
- Negative controls/context: AMD Radeon launches, US debt-ceiling headlines,
  broad market, TSMC and unrelated lifestyle headlines.

A URL path is a **hint**, not a recovered article headline. Multiple matching
URL-derived hints may be grouped as **possible reposts** but are never
considered the same event without article-text or independently corroborated
source verification. In the event research packet, no hints become independent
human Gold, guaranteed open-time facts, verified issuers, or price-causal labels.

### Separate primary evidence

NVIDIA official Q1 FY2024 results:
https://investor.nvidia.com/news/press-release-details/2023/NVIDIA-Announces-Financial-Results-for-First-Quarter-Fiscal-2024/

The company disclosed on 2023-05-24: quarterly revenue **$7.19 billion** and
Q2 revenue outlook **$11.0 billion**. Independently, its advance call notice
https://nvidianews.nvidia.com/news/nvidia-sets-conference-call-for-first-quarter-financial-results-6891941
scheduled the 2023-05-24 results announcement at *approximately* 1:20 pm
Pacific (4:20 pm US Eastern), and its earnings call at 2 pm Pacific.
**Scheduled timing is NOT an independently verified publication timestamp**,
so this implementation does not assign `release_at` or mark the disclosure
tradable at a specific minute. It is a *plausible earnings-guidance catalyst*,
not a quantified causal attribution to the next day's +24.37% NVDA return.

## Implementation and Mac commands

The existing `historical-story` command consumes the already
collected private GDELT JSON; no new web/news request or model call occurs
during `--prepare`.

```sh
bash scripts/news-learning-review.sh --update historical-story \
  --scout "$HOME/Library/Application Support/TrisMarketEventIntelligence/nvda-gdelt-gkg-rank1.json" \
  --prepare \
  --output "$HOME/Library/Application Support/TrisMarketEventIntelligence/nvda-story-review-rank1.json"
```

The private created report carries the GKG packet SHA256, all distinct URL
references, URL-only category *hints*, conservative repost candidates,
original archive UTC slot, missing article publication clocks, and the
separately attributed NVIDIA press-release reference. It reports source
coverage and unverified status without making up article text or macro
indicators.

For a **separate explicit** optional local Gemma retrospective hypothesis
using *at most 12 URL metadata rows*, with the existing injected provider:

```sh
bash scripts/news-learning-review.sh historical-story \
  --scout "$HOME/Library/Application Support/TrisMarketEventIntelligence/nvda-gdelt-gkg-rank1.json" \
  --run --model gemma4:e2b --limit 12 \
  --output "$HOME/Library/Application Support/TrisMarketEventIntelligence/nvda-story-gemma-hypothesis-rank1.json"
```

This does **not** send the full event study prices to the model. It supplies
source URL-derived hints and the separately cited primary statement. Model
returns a structured *unverified hypothesis* with source reference IDs,
plausible conditional mechanism, source limitations, uncertainty and
`causal_effect_proven: false`. Python checks structure, exact reference IDs,
output size and non-promotion; **semantic truth remains unverified**. A
valid JSON response does not certify the explanation.

Results use private create-only mode-0600 output; change file name on
retries. No training occurs; no news source/LLM is scheduled or activated
24/7. Existing final-test and production boundaries are untouched.

## Next research evolution

1. Verify actual publication timestamps and permissible original article
   text for candidate URLs, with errors and 404s tracked as missing evidence.
2. Use URL hints only to prioritize publisher/issuer documents. Deduplicate
   verified text/attribution and group real events, not noisy mere mentions.
3. Attach actual FRED/ALFRED economic values only when historical vintage
   **and** public release/availability timestamps are grounded.
4. Allow Gemma to compare primary/secondary dated source contents and
   simultaneous confounders, keeping its causal assessment probabilistic.
5. Expand to TOP100 + comparison-day samples and evaluate genuinely unseen
   forward predictions with time-based embargo, rather than scoring
   retrospective explanations as model forecast performance.
