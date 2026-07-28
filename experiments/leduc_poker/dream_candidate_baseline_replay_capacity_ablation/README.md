# DREAM candidate learned-baseline replay-capacity ablation (Experiment 26)

This experiment varies only `baseline_memory_capacity` around the
architecture-selected DREAM candidate. It tests whether the learned
control-variate networks benefit from fresher replay when the sampling policy
and regret estimates are changing over training.

The default arms are `1e6`, `5e5`, and `1e5` learned-baseline-memory entries.
All other settings are fixed, including linear average-strategy weighting and
`epsilon=0.06`.

## Run

```bash
python -m experiments.leduc_poker.dream_candidate_baseline_replay_capacity_ablation.run
```

Full runs reuse the tracked Experiment 22 candidate baseline comparator by default. Add `--train-baseline` only when you intentionally want to retrain that comparator.

## Smoke test

```bash
python -m experiments.leduc_poker.dream_candidate_baseline_replay_capacity_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants candidate_baseline_memory_1m_baseline,candidate_baseline_memory_100k \
  --train-baseline \
  --output-root outputs/smoke_tests/dream_candidate_baseline_replay_capacity_ablation
```

## GCP Batch smoke test

```bash
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp26-baseline-replay-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_baseline_replay_capacity_ablation.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --variants candidate_baseline_memory_1m_baseline,candidate_baseline_memory_100k \
    --train-baseline \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_baseline_replay_capacity_ablation" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"
```

Primary plots include `dream_candidate_baseline_replay_capacity_exploitability_by_nodes.png`.
