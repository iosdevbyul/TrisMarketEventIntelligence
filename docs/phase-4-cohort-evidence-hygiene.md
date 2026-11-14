# Stage 4 — source uniqueness and evidence sufficiency for new v8 pilots

The first seven Gemma 2 v8 checkpoints were all NO_HYPOTHESIS. Source review
showed two obvious data problems: producer-price-index entries #11 and #19
contained only a release title, and #35/#36 represented the same Microsoft
outage from different BBC feeds. Events #23/#26 contained numeric financial
information, so their abstention is still **unresolved**; we must not claim
that v8 is correct or broken based solely on checkpoint counts.

For `--live-next` (and **only** for the never-before-attempted candidate
selection), this update:
- Excludes exact duplicate article URLs after removing tracking parameters.
  Meaningful identifiers (e.g. `nttId`) stay in the URL.
- Excludes identical longer source excerpts even when feed URLs differ.
- Excludes empty evidence and title-only Korean monthly producer-price-index
  notices with no reported index or change figure.
- Excludes copies of events already attempted with the same model, even when
  the earlier attempt used another prompt version.
- Reports `skipped_duplicate_sources` and `skipped_sparse_evidence` as
  descriptive candidate-selection counts in the saved preview.

No event is deleted or reclassified; past v6/v7/v8 hypotheses, review files
and checkpoints remain unchanged. These are narrow screening heuristics, not
source credibility tests. The entire corpus is **not** deduplicated and
unverified information is **not** promoted to verified.

Mac opt-in inference (up to 3 *new* selected IDs, only after tests pass):

```sh
bash scripts/verify.sh --update --live-next 3
```

Offline checks with no Ollama invocation:

```sh
bash scripts/verify.sh --update
```

Review the resulting private `cohort-review-*.json` and the normal
`verify-*.json` before changing the prompt. The remaining hypothesis
abstentions on meaningful financial metrics need human/source review rather
than an unconditional requirement that the model generate an impact.
