# Learning-08 — read-only Mac Gemma 4 capacity preflight

This command is an **opt-in observation**, not a model download, runtime
benchmark, news inference or approval to change the running collector.
It checks Mac physical RAM, logical CPU count, architecture, a single
`vm_stat` snapshot, free disk space and loopback-only Ollama
`GET /api/tags` and `GET /api/ps`. All OS calls use specific
non-shell, read-only utility arguments. No API keys, home-directory paths
or private source news appear in the report.

Run via the existing Mac shell workflow when actual hardware diagnosis
is necessary (not required for CI):

```sh
bash scripts/news-learning-review.sh --update gemma-mac-preflight \
  --gemma4 gemma4:e4b \
  --output "$HOME/Library/Application Support/TrisMarketEventIntelligence/new-gemma4-hardware-check.json"
```

`--output` is optional. If used, a **new** owner-only private JSON file
is created; it never overwrites a prior audit. The underlying commands
never call `ollama pull`, `ollama run`, model inference, launchd,
production SQLite or non-loopback network APIs.

## Reading the result

- `UNSUPPORTED_HOST_NOT_MACOS`: no Mac-specific sampling attempted.
- `LOCAL_OLLAMA_UNAVAILABLE`: local Ollama is unreachable or returned
  an invalid response. The program does not start it.
- `LOCAL_MODEL_NOT_INSTALLED`: at least one exact named model tag
  does not appear installed. This is not permission to download it.
- `MEMORY_HEADROOM_LOW_FOR_SHARED_MAC`: a conservative instantaneous
  reading (free + inactive + speculative pages) is below the 4 GiB
  shared-Mac reserve or total RAM itself is too low.
- `CAPACITY_NOT_VERIFIED`: installed models and observed memory have
  no obvious preflight blocker, **not** a claim the model will run.

Physical RAM is not available RAM; inactive pages are only a snapshot
and may not be reclaimable when inference starts. Gemma model download
size is **not loaded model RSS, KV-cache size or total inference memory**.
No model can be certified to fit from these numbers; actual latency,
OS pressure under the news collector, memory peak and multi-model
swap costs require an explicitly approved, short, guarded real
inference trial. Be conservative on Intel Macs and shared 24/7 hosts.

The report always sets `realInferenceMemoryFitVerified=false`,
`newsModelQualityVerified=false`, `modelDownloadAttempted=false`,
and `productionModelChanged=false`. No human-reviewed NEWS metrics
or stock-market price data are involved. Gemma 2 remains in place.
