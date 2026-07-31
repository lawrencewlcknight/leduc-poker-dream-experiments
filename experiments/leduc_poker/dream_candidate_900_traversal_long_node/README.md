# DREAM candidate 900-traversal long-node run (Experiment 35)

This experiment trains the Experiment 22 architecture-selected DREAM candidate
with `900` outcome-sampling traversals per player per iteration. The run uses
`1,300` DREAM iterations, which gives approximately the same traversal-unit
budget as the existing `160`-traversal, `7,300`-iteration long-node run and is
intended to touch roughly `15m` nodes per seed.

The default run preserves the Experiment 22 candidate settings: five seeds,
`epsilon=0.06`, policy and learned-baseline networks `[32, 32]`, advantage
networks `[128, 128]`, linear average-strategy weighting, policy training every
`25` iterations, and learned-baseline training every iteration.

## Run

```bash
python -m experiments.leduc_poker.dream_candidate_900_traversal_long_node.run
```

## Smoke test

```bash
python -m experiments.leduc_poker.dream_candidate_900_traversal_long_node.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests/dream_candidate_900_traversal_long_node
```

## GCP Batch smoke test

```bash
./gcp/submit_batch_experiment.sh \
  "ld-exp35-t900-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_900_traversal_long_node.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_900_traversal_long_node" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"
```

Primary plots include
`dream_candidate_900_traversal_long_node_exploitability_by_nodes.png` and
`dream_candidate_900_traversal_long_node_average_policy_value_by_nodes.png`.
