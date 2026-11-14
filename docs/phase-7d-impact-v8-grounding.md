# Stage 4 / Phase 7-D: Evidence-to-market reasoning (v8)

This revision is motivated by a **real Gemma 2 v7** result from economic event 13:
an UNVERIFIED report that listed improvements to external-audited firms' growth,
profitability, and stability. The saved GLOBAL/BROAD_MARKET rationale was:
"Improved financial performance may indicate broader economic strength."

The original text supported the reported metric improvement, but the rationale
did not explain **how** investors, valuations, earnings expectations or market
pricing might react. The single source excerpt is not proof that global equity
markets strengthened. The v7 analysis remains part of the immutable history.

## New behavior

- Prompt v8 explicitly separates a source fact, a conditional market-facing
  *transmission mechanism*, expected exposure/market and its limitations.
- Do not treat quoted financial metrics as a measured market reaction.
- Do not infer a global trend or a macro surprise from one report.
- Only include drivers if defensible from the source and explicitly described
  as hypothetical mechanisms; an empty list is allowed.
- No evidence or no defensible exposure means an empty `impacts` list.
- For BROAD_MARKET hypotheses, a deterministic *narrow* lexical guard
  rejects generic economic summaries that mention no market transmission.
  It does **not** independently verify causal truth, factuality or direction.
  Exact evidence and UNVERIFIED=>HIGH checks remain mandatory.
- Model+prompt-version checkpointing isolates all v8 executions from v6/v7.
  Prior results and error history are not rewritten.

## Mac manual validation

```sh
git switch main
git pull --ff-only origin main
python -m pip install -e ".[dev]"
python -m pytest -q tests/test_impact_v8.py
python -m pytest -q
export MARKET_EVENT_LLM_MODEL=gemma2:latest
market-event-analyze --pilot-preview
```

After reviewing the candidate IDs, **explicitly** run up to three selected
events. For a controlled comparison with the formerly saved v7 result, event
13 can be explicitly selected because v8 has a different versioned checkpoint:

```sh
market-event-analyze --provider local --event-ids 13 \
  --batch-size 1 --timeout-seconds 300
market-event-pilot-report --event-ids 13 \
  --model-id gemma2:latest
market-event-pilot-report --event-ids 13 \
  --model-id gemma2:latest --prompt-version market-impact-v7
market-event-analyze --show-errors
```

Successful v8 generation does not mean the causal hypothesis is true.
Compare actual rationale and literal evidence, then use a fresh independent
human review sheet; do not overwrite the preexisting audit CSV.
