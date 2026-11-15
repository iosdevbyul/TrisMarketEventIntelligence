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

## Local Ollama inference budget for historical research

An actual 2026-11-15 Mac run of `historical-multi --run` with the
44-reference NVIDIA scout failed with
`historical-multi-research: MODEL_GENERATION_TRUNCATED`.
The Ollama transport explicitly reported `done_reason=length`.
The model output was **not** a complete JSON research artifact and
was not saved or used as training, Gold, or a hypothesis.

This research CLI now explicitly requests `num_ctx=8192` and
`num_predict=3072` from local Ollama (thinking off), allowing
space for three competing structured hypotheses and full 44-source
context; these limits are **per-command**, not new global defaults.
For a constrained host the operator may select
`--context-tokens 4096..16384` and
`--output-tokens 1024..4096` explicitly. Higher values increase
local CPU/memory use and do not guarantee a valid model response.
The same validated research schema and source-citation constraints apply.

```sh
bash scripts/news-learning-review.sh --update historical-multi \
  --scout "$HOME/Library/Application Support/TrisMarketEventIntelligence/nvda-gdelt-gkg-rank1.json" \
  --run --model gemma4:e2b \
  --output "$HOME/Library/Application Support/TrisMarketEventIntelligence/nvda-multi-hypotheses-rank1-v3.json"
```

If the provider still reports `MODEL_GENERATION_TRUNCATED`, do not
discard constraints, infer a partial output, or claim improvement.
Investigate local `num_ctx` vs actual token usage and, if necessary,
split independent topic-group analyses followed by a separate
source-cited comparison. This command never downloads a larger model.

## Learning 06: deterministic historical quality audit

Actual Gemma4:e2b 44-reference NVIDIA research from 2026-11-15
completed with three competing hypotheses, two local model calls, and
`RETROSPECTIVE_HYPOTHESES_UNVERIFIED`. This proves a completed inference,
not accurate explanation or forecast skill. The model used the category
`UNASSESSED` with `PRIMARY_DOCUMENT`, topic labels instead of mechanisms,
and causal claims such as "primary driver" without independent attribution.

Run **without another model call** against the existing PRIVATE output:

```sh
bash scripts/news-learning-review.sh --update historical-quality \
  --research "$HOME/Library/Application Support/TrisMarketEventIntelligence/nvda-multi-hypotheses-rank1-v5.json" \
  --output "$HOME/Library/Application Support/TrisMarketEventIntelligence/nvda-research-quality-v1.json"
```

The new quality audit preserves the unverified status and writes a new
private file only. It flags category/evidence-strength conflicts,
evidence references inconsistent with strength, label-only economic
mechanisms, and selected categorical causal statements. It counts
**explicitly cited** unique archive references, not text that was read.
Detection is intentionally heuristic; false negatives are possible.
It does not certify source truth, timestamp validity, article rights,
macro vintages, real causality, predictive value, or model accuracy.

The report has `suggestedMethodLessons` as **unapproved proposals**.
`lessonsApproved=false`, no Gold, no production writes, no
autonomous promotion, and no model-weight changes. Never convert
proposals into `reviewedByHuman=true` automatically. A person must
inspect and approve the original evidence and then independently create
a private `human_reviewed_research_lessons_v1` file using the
documented optional `--reviewed-lessons` path. Evaluation of improved
behavior requires separate held-out cases rather than repeating only
the same NVIDIA example.

## Learning 06: explicit analyst review of research findings

After `historical-quality` successfully produces a private audit, prepare a
human-review template. Preparation never accepts or trains anything:

```sh
bash scripts/news-learning-review.sh --update historical-human-review --prepare \
  --audit "$HOME/Library/Application Support/TrisMarketEventIntelligence/nvda-research-quality-v2.json" \
  --output "$HOME/Library/Application Support/TrisMarketEventIntelligence/nvda-human-review-template-v1.json"
```

The template starts with `status=AWAITING_HUMAN_REVIEW`,
`sourceEvidenceReviewed=false`, `approved=false`, every finding decision
`PENDING`, and every proposed lesson `selected=false`. No approved
lesson file is created at this stage. The analyst must inspect independent
sources and explicitly edit the private template: set a reviewer identifier,
`sourceEvidenceReviewed=true`, `approved=true`, decide every finding
(`CONFIRMED`, `REJECTED`, or `UNCERTAIN`) and enter a specific
`reviewNote` for each. Select only lessons supported by independently
inspected evidence and add at least 30 characters of `independentEvidence`
for each selected lesson. No evidence may be added to unselected lessons.

Only after that manual work, the analyst can explicitly attest and export:

```sh
bash scripts/news-learning-review.sh historical-human-review --finalize \
  --audit "$HOME/Library/Application Support/TrisMarketEventIntelligence/nvda-research-quality-v2.json" \
  --review "$HOME/Library/Application Support/TrisMarketEventIntelligence/nvda-human-review-template-v1.json" \
  --reviewer HUMAN_IDENTIFIER --confirm-human-review \
  --output "$HOME/Library/Application Support/TrisMarketEventIntelligence/nvda-reviewed-lessons-v1.json"
```

The implementation verifies the original audit digest, unchanged finding
identities, explicit decisions and notes, lesson identities, independent
evidence, and reviewer confirmation; it never silently sets human approval.
This is an attestation mechanism, not cryptographic proof that a human
actually looked at sources. Keep review and approved lessons private, never
commit them to Git. The approved output is compatible with the existing
`historical-multi --reviewed-lessons` option but remains a procedural
memory hint, never independent Gold, ground truth, weight fine-tuning,
permission to trade, or proof of generalization. Evaluate improvements on
independent cases before making claims of better research quality.

## Learning 06: reproducible baseline and reviewed-feedback comparison

A verified 44-reference NVIDIA historical research result and its deterministic
quality audit can now be captured **without another Ollama call**:

```sh
bash scripts/news-learning-review.sh --update historical-learning-compare \
  --baseline-research "$HOME/Library/Application Support/TrisMarketEventIntelligence/nvda-multi-hypotheses-rank1-v5.json" \
  --baseline-audit "$HOME/Library/Application Support/TrisMarketEventIntelligence/nvda-research-quality-v2.json" \
  --output "$HOME/Library/Application Support/TrisMarketEventIntelligence/nvda-learning-baseline-v1.json"
```

The command deterministically recomputes the audit from the model artifact,
checks that the private original audit matches, and records source SHA256,
per-finding counts, source citation count and reference count. This only
records a baseline; it does not mark findings as true, approve lessons,
or retrain a model.

After **actual independent human review** produces the approved lessons file
and a separately generated Gemma research result using
`historical-multi --reviewed-lessons`, separately audit the new research
with `historical-quality`. Then run:

```sh
bash scripts/news-learning-review.sh historical-learning-compare \
  --baseline-research BASELINE_RESEARCH_JSON --baseline-audit BASELINE_AUDIT_JSON \
  --candidate-research COACHED_RESEARCH_JSON --candidate-audit COACHED_AUDIT_JSON \
  --reviewed-lessons HUMAN_APPROVED_LESSONS_JSON --output NEW_PRIVATE_COMPARISON_JSON
```

The pair must match model ID, prompt version, input research packet digest,
reference cohort count, and the exact SHA256 of the accepted human lessons.
Mismatched or tampered reports are refused. The output reports differences
in heuristic findings by code, rather than falsely claiming higher model
accuracy. **Comparisons of the same historical example are diagnostic only**.
An independent set of dates and news cases, separately reviewed and kept
out of the lesson-design process, is needed before a generalization claim.
No prediction accuracy or causal price effect is proven here; no Gold or
production auto-promotion occurs.

The existing reviewed-lessons reader now accepts the optional
`reviewer`, `auditSha256`, and `researchSha256` lineage fields emitted
by the human review CLI; older minimal human-reviewed lesson files still
work. This fixes the actual handoff contract, not the quality score.

The `--update` script now restarts the latest version of itself after
`git pull --ff-only`, preventing new subcommands from unexpectedly printing
the old usage instructions during the very first updated invocation.

## Learning 06: independent historical NEWS case cohort (2023, no prices)

Three **new** historical events were sourced from publicly reachable original
issuer disclosures and separately pinned in
`research/independent-historical-news-2023.json`:

| Company | Original disclosure date | Issuer/SEC URL | Confirmed evidence |
| --- | --- | --- | --- |
| Microsoft | 2023-07-25 | https://www.microsoft.com/en-us/investor/earnings/fy-2023-q4/press-release-webcast | Revenue USD 56.189bn; Azure +26% reported YoY; Microsoft Cloud USD 30.3bn |
| Alphabet | 2023-07-25 | https://www.sec.gov/Archives/edgar/data/1652044/000165204423000067/googexhibit991q22023.htm | Revenue USD 74.604bn; Google Cloud USD 8.031bn vs 6.276bn; Cloud operating profit USD 395m vs a USD 590m loss |
| AMD | 2023-08-01 | https://www.amd.com/en/newsroom/press-releases/2023-8-1-amd-reports-second-quarter-2023-financial-results.html | Revenue USD 5.359bn; GAAP operating loss USD 20m; planned Q4 2023 MI300 launch and ramp |

The nine statements are **short factual paraphrases** of verified source
disclosures, not scraped copyrighted articles, synthetic GDELT references,
unverified publisher clocks, model training labels, or claims about stock
returns. A disclosure date is not independently verified intraday
availability. These issuer documents do not provide independent sector news;
unsupported alternative explanations must remain tentative.

Three actual issuer releases are **three primary sources**, not a 44-article
GDELT sample. Existing NVIDIA 2023-05-25 lessons and human-review template
must stay untouched. This independent challenge is kept separate from
NVIDIA's retrospectively chosen case and does not use price or return inputs.

Inspect exact source inventory with no model calls:

```sh
bash scripts/news-learning-review.sh --update historical-independent \
  --cohort research/independent-historical-news-2023.json --preview
```

Opt in to three real locally installed Ollama Gemma4:e2b research calls,
with no approved NVIDIA lessons:

```sh
bash scripts/news-learning-review.sh historical-independent \
  --cohort research/independent-historical-news-2023.json \
  --run --model gemma4:e2b \
  --output "$HOME/Library/Application Support/TrisMarketEventIntelligence/independent-news-research-v1.json"
```

Only after a completed output, evaluate provenance and structural safeguards:

```sh
bash scripts/news-learning-review.sh historical-independent \
  --cohort research/independent-historical-news-2023.json \
  --evaluate \
  --research "$HOME/Library/Application Support/TrisMarketEventIntelligence/independent-news-research-v1.json" \
  --output "$HOME/Library/Application Support/TrisMarketEventIntelligence/independent-news-evaluation-v1.json"
```

The evaluator checks exact case/cohort digest, all nine factual-source
references, three hypothesis types per case, source IDs, mandatory
noncausality and evidence-limitation fields. Heuristic flags identify
unsupported sector source citations and overconfident market attribution.
These are **mechanical attribution checks**, not verified independent
semantic correctness. Real-case Gemma performance and cross-case
generalization remain **unverified until actual Mac inference and human
inspection**. There is no auto-approved Gold, model-weight training,
promotion, or investment execution. Comparisons with future approved
lessons must not train on or preselect this independent evaluation cohort.
