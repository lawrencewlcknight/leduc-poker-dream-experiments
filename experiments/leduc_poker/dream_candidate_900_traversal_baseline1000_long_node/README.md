# DREAM candidate 900-traversal baseline-1000 long-node run (Experiment 39)

This experiment starts from the Experiment 22 architecture-selected DREAM
candidate and changes two training-budget settings:

- `num_traversals=900` outcome-sampling traversals per player per iteration.
- `baseline_network_train_steps=1000` learned-baseline minibatches per player
  per iteration.

The learned baseline is trained every DREAM iteration, as in the Experiment 22
configuration. The run uses the vectorized learned-baseline implementation, keeps
`epsilon=0.06`, and trains for `1,300` iterations, targeting roughly `15m` nodes
touched per seed. The default run uses three seeds: `1234`, `2025`, and `31415`.

## Run

```bash
python -m experiments.leduc_poker.dream_candidate_900_traversal_baseline1000_long_node.run
```

## Smoke test

```bash
python -m experiments.leduc_poker.dream_candidate_900_traversal_baseline1000_long_node.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests/dream_candidate_900_traversal_baseline1000_long_node
```

## GCP Batch smoke test

```bash
./gcp/submit_batch_experiment.sh \
  "ld-exp39-t900b1000-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_900_traversal_baseline1000_long_node.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_900_traversal_baseline1000_long_node" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"
```

Primary plots include
`dream_candidate_900_traversal_baseline1000_long_node_exploitability_by_nodes.png`
and
`dream_candidate_900_traversal_baseline1000_long_node_average_policy_value_by_nodes.png`.
