# DREAM candidate long-node run (Experiment 41)

This experiment is the dedicated single-arm version of the Experiment 22
architecture-selected DREAM candidate trained to the long-node budget. Experiment
33 already includes this configuration as its `epsilon=0.06` arm, but also
trains the `epsilon=0.20` treatment. This experiment avoids that additional
compute when the desired run is simply the Experiment 22 candidate over roughly
`15m` nodes.

The configuration uses the vectorized learned-baseline implementation and keeps
the Experiment 22 candidate settings:

- `epsilon=0.06`.
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
python -m experiments.leduc_poker.dream_candidate_long_node_run.run
```

## Smoke test

```bash
python -m experiments.leduc_poker.dream_candidate_long_node_run.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests/dream_candidate_long_node_run
```

## GCP Batch smoke test

```bash
./gcp/submit_batch_experiment.sh \
  "ld-exp41-longnode-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_long_node_run.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_long_node_run" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"
```

Primary plots include
`dream_candidate_long_node_exploitability_by_nodes.png` and
`dream_candidate_long_node_average_policy_value_by_nodes.png`.
