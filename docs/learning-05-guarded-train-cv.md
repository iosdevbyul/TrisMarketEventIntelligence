# Learning-05 — opt-in guarded TRAIN-only news CV

The existing NEWS-only process guard now supports an explicit `--research-mode train-cv` mode, in addition to its unchanged default `legacy` loop. The new mode invokes the existing 18-configuration TRAIN-fold CV researcher inside the guarded child process, without adding a new training engine or modifying the production model.

## Run manually on Mac (not installed or auto-started)

```sh
bash scripts/news-learning-review.sh --update guard-research --run \
  --research-mode train-cv --max-iterations 18 \
  --cohort /private/news-cohort.json \
  --holdout /private/news-holdout.json \
  --review /private/human-review.json \
  --journal "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-research.sqlite3" \
  --status-file "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-guard-status.json" \
  --lock-file "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-guard.lock"

bash scripts/news-learning-review.sh guard-research --status \
  --status-file "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-guard-status.json"

bash scripts/news-learning-review.sh train-cv --audit \
  --journal "$HOME/Library/Application Support/TrisMarketEventIntelligence/news-research.sqlite3"
```

The guard enforces child CPU RLIMIT and wall time, samples the direct child RSS (not a kernel-enforced memory quota), and cooperatively respects a supplied stop flag via its process monitor. Research is still finite: no more than 18 attempts per batch, durable journal deduplication across restarts, and a truthful stop when the finite space is exhausted or labels are insufficient. In TRAIN CV mode the child does not check the stop flag between individual CV experiments, but the parent can terminate its process group.

A guard `COMPLETED` status means the child exited with code zero; it is **not** proof that learning improved, that human labels are sufficient, or that an experiment ran. Use `train-cv --audit` for journal evidence and the independent human review protocol before making any performance claim.

Original NEWS inputs and the original legacy research journal are not rewritten. Validation/TEST gold is never used for CV selection. No production promotion, launchd installation, 24/7 activation, Gemma 4 installation, stock price, OHLCV, trading volume, returns, or DonghakStockVision data access is involved. Synthetic CI only; real Mac execution and independent human evaluation remain pending.
