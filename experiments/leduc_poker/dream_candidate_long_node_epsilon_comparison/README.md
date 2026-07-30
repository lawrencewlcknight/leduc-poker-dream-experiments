# DREAM candidate long-node epsilon comparison (Experiment 33)

This experiment tests the architecture-selected DREAM candidate under a much
larger fixed node budget. It compares the current candidate before the
candidate-parameter ablations against the same configuration with only
`epsilon` changed to `0.20`.

Both arms are trained over five seeds. The long run keeps `160` traversals per
player per DREAM iteration and increases training to `7,300` iterations, which
is expected to produce roughly `15m` nodes touched per seed based on the
observed node rate of the candidate configuration.

The baseline arm uses policy and learned-baseline networks `[32, 32]`,
advantage networks `[128, 128]`, linear average-strategy weighting,
`epsilon=0.06`, `100` policy steps, policy training every `25` iterations,
advantage batch size `1024`, and learning rate `0.003`.

## Run

```bash
python -m experiments.leduc_poker.dream_candidate_long_node_epsilon_comparison.run
```

This experiment intentionally trains both arms. It does not reuse the tracked
Experiment 22 baseline artifact.

## Smoke test

```bash
python -m experiments.leduc_poker.dream_candidate_long_node_epsilon_comparison.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests/dream_candidate_long_node_epsilon_comparison
```

## GCP Batch smoke test

```bash
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp33-long-node-epsilon-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_long_node_epsilon_comparison.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_long_node_epsilon_comparison" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"
```

Primary plots include
`dream_candidate_long_node_epsilon_exploitability_by_nodes.png` and
`dream_candidate_long_node_epsilon_average_policy_value_by_nodes.png`.
