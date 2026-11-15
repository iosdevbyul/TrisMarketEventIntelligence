# Historical economic-event research: teach Gemma a method, not an answer

This is a **standalone** TrisMarketEventIntelligence workflow. It combines
10 years of actual NVDA/SPY adjusted prices and retrospective TOP100 market
shock candidates with the **44 archived GDELT metadata references** found
for 2023-05-25. It preserves all 44 references in a single comparison
prompt, grouped by **unverified** URL-derived category (rather than only
the first 12). An independent, dated NVIDIA Q1 FY2024 issuer release is
provided separately as documentary evidence, and missing economic-indicator
vintages are explicitly marked missing.

## What Gemma can learn here

- **In-context coaching**: a versioned research-method prompt defines
  source priority, evidence limitations, alternative explanations, and
  examples of good and bad reasoning. The original fixed model weights are
  not modified by the conversation.
- **Human-reviewed procedural memory**: an optional, separately pinned
  JSON file contains genuinely human-reviewed method lessons. The model
  reads these as guidelines on future runs; its own output is never
  automatically treated as a human review.
- **Weight fine-tuning**: NOT implemented or implied. It requires a separate
  rights-cleared training set, appropriate hardware/training stack, held-out
  objective evaluation, consent and model license review. Do not start
  fine-tuning or install new weights automatically.

## Correct comparison method

1. Review ALL available GKG reference candidates (not first 12 only).
2. Group by subject/possible syndication, but distinguish URL slug hints
   from read-and-verified article bodies.
3. Give a dated first-party earnings disclosure its own evidence type.
   Higher documentary weight does **not** make it automatically causal.
4. Propose three competing explanations with exact evidence IDs:
   company fundamentals, competitor/sector alternative, and insufficient
   or macro/market alternative where evidence is missing.
5. Explain how the event *might* change expected business prospects,
   explicitly address confounders and unverified news/publication timestamps.
6. Check source IDs and required uncertainty constraints in Python.
   No invented source IDs, missing first-seen timestamp, direct attribution
   of precise market returns, or retrospective explanation as forecast gold.
7. Compare a sealed chronological forward-replay set independently before
   claiming any useful predictive skill.

Existing first Gemma response on the initial 12 URLs:
`historical_story_hypothesis_v1`, `gemma4:e2b`, four cited URLs,
competitor/sector (AMD and GPU rivalry) only. Its own `causal_effect_proven`
was false; there was no actual training or measured forecasting benefit.
This new workflow enables competing event hypotheses and full-reference
coverage. It does NOT force the earnings hypothesis to win.

## Mac usage

Use the already acquired rank-1 GKG scout. For safe preflight (NO model
call, no remote download):

```sh
bash scripts/news-learning-review.sh --update historical-multi \
  --scout "$HOME/Library/Application Support/TrisMarketEventIntelligence/nvda-gdelt-gkg-rank1.json" \
  --preview \
  --output "$HOME/Library/Application Support/TrisMarketEventIntelligence/nvda-multi-research-preflight.json"
```

Then explicitly ask local Ollama Gemma4:e2b for three candidate
explanations. It uses **one** bounded inference call, no model
installation/fine-tuning:

```sh
bash scripts/news-learning-review.sh historical-multi \
  --scout "$HOME/Library/Application Support/TrisMarketEventIntelligence/nvda-gdelt-gkg-rank1.json" \
  --run --model gemma4:e2b \
  --output "$HOME/Library/Application Support/TrisMarketEventIntelligence/nvda-multi-hypotheses-rank1.json"
```

### Optional human-reviewed lessons

After actually reviewing evidence and model errors, the operator may
create a **private** 0600 local file, never place it inside Git. Its
structure is:

```json
{
  "schemaVersion": "human_reviewed_research_lessons_v1",
  "reviewedByHuman": true,
  "lessons": [
    {
      "lesson": "Compare issuer guidance to competition coverage before ranking.",
      "evidence": "The analyst independently inspected dated issuer evidence."
    }
  ]
}
```

Do NOT set `reviewedByHuman=true` merely because the model generated a
critique. The reviewer must independently evaluate the lesson. Max six
lessons, 300 characters per field, and an input SHA256 is kept in the
result. Then pass `--reviewed-lessons FILE` to a new `historical-multi
--run` invocation with a **new output path**.

### Constraints and current truth

The current scout does **not** include licensed full-text articles, verified
first publisher publication timestamps, contemporaneous CPI/payroll/Fed
vintages with minute-level availability, or an objectively sealed
predictive backtest. Every model result remains an *unverified*
retrospective explanation. `weightTrainingPerformed=false`,
`forecastImprovementMeasured=false`, and `promotionAllowed=false`.
Neither independent Gold nor new market data are written by this
workflow. Historic TOP100 membership is outcome-conditioned and must
not enter future time-point predictor features. Mac worker/24-hour
learning is not activated.
