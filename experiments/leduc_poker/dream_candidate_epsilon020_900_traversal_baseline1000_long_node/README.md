# DREAM epsilon-0.20 900-traversal baseline-1000 long-node run (Experiment 40)

This experiment keeps the Experiment 39 paper-style DREAM configuration and
changes only the exploration rate to `epsilon=0.20`.

The run therefore starts from the Experiment 22 architecture-selected candidate
and uses:

- `epsilon=0.20`.
- `num_traversals=900` outcome-sampling traversals per player per iteration.
- `baseline_network_train_steps=1000` learned-baseline minibatches per player
  per iteration.

The learned baseline is trained every DREAM iteration, as in Experiment 39. The
run uses the vectorized learned-baseline implementation and trains for `1,300`
iterations, targeting roughly `15m` nodes touched per seed. The default run uses
three seeds: `1234`, `2025`, and `31415`.

## Run

```bash
python -m experiments.leduc_poker.dream_candidate_epsilon020_900_traversal_baseline1000_long_node.run
```

## Smoke test

```bash
python -m experiments.leduc_poker.dream_candidate_epsilon020_900_traversal_baseline1000_long_node.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests/dream_candidate_epsilon020_900_traversal_baseline1000_long_node
```

## GCP Batch smoke test

```bash
./gcp/submit_batch_experiment.sh \
  "ld-exp40-e02t900b1000-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_epsilon020_900_traversal_baseline1000_long_node.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_epsilon020_900_traversal_baseline1000_long_node" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"
```

Primary plots include
`dream_candidate_epsilon020_900_traversal_baseline1000_long_node_exploitability_by_nodes.png`
and
`dream_candidate_epsilon020_900_traversal_baseline1000_long_node_average_policy_value_by_nodes.png`.
