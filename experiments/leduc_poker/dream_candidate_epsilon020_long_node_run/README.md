# DREAM epsilon-0.20 candidate long-node run (Experiment 42)

This experiment replicates Experiment 41 and changes only the exploration rate
from `epsilon=0.06` to `epsilon=0.20`. It is a dedicated single-arm vectorized
run for the high-exploration version of the Experiment 22 architecture-selected
DREAM candidate over the long-node budget.

The configuration uses:

- `epsilon=0.20`.
- `num_traversals=160` outcome-sampling traversals per player per iteration.
- Policy and learned-baseline networks `[32, 32]`.
- Advantage networks `[128, 128]`.
- Linear average-strategy weighting.
- Policy training every `25` iterations for `100` minibatches.
- Advantage and learned-baseline training every iteration for `50` minibatches.

The run uses `7,300` DREAM iterations and five seeds, targeting roughly `15m`
nodes touched per seed.

## Run

```bash
python -m experiments.leduc_poker.dream_candidate_epsilon020_long_node_run.run
```

## Smoke test

```bash
python -m experiments.leduc_poker.dream_candidate_epsilon020_long_node_run.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests/dream_candidate_epsilon020_long_node_run
```

## GCP Batch smoke test

```bash
./gcp/submit_batch_experiment.sh \
  "ld-exp42-e02-longnode-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_epsilon020_long_node_run.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_epsilon020_long_node_run" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"
```

Primary plots include
`dream_candidate_epsilon020_long_node_exploitability_by_nodes.png` and
`dream_candidate_epsilon020_long_node_average_policy_value_by_nodes.png`.
