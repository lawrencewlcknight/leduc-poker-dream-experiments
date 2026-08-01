# DREAM candidate epsilon comparison with sparse baseline training at long-node horizon (Experiment 38)

This experiment takes Experiment 34 and extends it to the long-node training
horizon. It compares the architecture-selected DREAM candidate with
`epsilon=0.06` against the same configuration with `epsilon=0.20`, while both
arms train the learned baseline every `50` DREAM iterations.

The run uses the vectorized learned-baseline implementation. It keeps `160`
outcome-sampling traversals per player per DREAM iteration and increases
training to `7,500` iterations. Based on the observed Experiment 34 node rate,
this should touch roughly `15m` nodes per seed on average across the two arms.

The default run preserves Experiment 34's three-seed protocol and intentionally
trains both arms. It does not reuse a fixed Experiment 22 comparator artifact.

## Run

```bash
python -m experiments.leduc_poker.dream_candidate_epsilon_baseline50_long_node_comparison.run
```

## Smoke test

```bash
python -m experiments.leduc_poker.dream_candidate_epsilon_baseline50_long_node_comparison.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests/dream_candidate_epsilon_baseline50_long_node_comparison
```

## GCP Batch smoke test

```bash
./gcp/submit_batch_experiment.sh \
  "ld-exp38-eb50ln-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_epsilon_baseline50_long_node_comparison.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_epsilon_baseline50_long_node_comparison" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"
```

Primary plots include
`dream_candidate_epsilon_baseline50_long_node_exploitability_by_nodes.png` and
`dream_candidate_epsilon_baseline50_long_node_average_policy_value_by_nodes.png`.
