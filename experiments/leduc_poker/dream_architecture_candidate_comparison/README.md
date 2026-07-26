# DREAM architecture-candidate comparison (Experiment 22)

This experiment compares the original DREAM baseline architecture with the best
candidate selected from the architecture ablations. It is intended as a
confirmation-style matched comparison rather than another broad architecture
screen.

The paired variants are:

- `baseline_all_2x32`: policy, advantage, and learned-baseline networks all use
  `[32, 32]`;
- `candidate_advantage_2x128_policy_baseline_2x32`: the average-policy and
  learned-baseline networks remain `[32, 32]`, while the player-specific
  advantage networks use `[128, 128]`.

The default run uses the five baseline seeds `1234`, `2025`, `31415`, `27182`,
and `16180`, with the same training protocol as the DREAM baseline: `175`
iterations, `160` traversals per player per iteration, exact exploitability
evaluation every `25` iterations, `epsilon=0.06`, replay capacity `1e6`, and
intermittent average-policy training every `25` iterations.

The primary comparison metrics are final exploitability, final-window
exploitability, exploitability AUC by nodes touched, final policy-value error,
and paired seed-level deltas against the baseline.

## Run

```bash
python -m experiments.leduc_poker.dream_architecture_candidate_comparison.run
```

## Smoke test

```bash
python -m experiments.leduc_poker.dream_architecture_candidate_comparison.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants baseline_all_2x32,candidate_advantage_2x128_policy_baseline_2x32 \
  --output-root outputs/smoke_tests/dream_architecture_candidate_comparison
```

## GCP Batch smoke test

```bash
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp22-architecture-candidate-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_architecture_candidate_comparison.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --variants baseline_all_2x32,candidate_advantage_2x128_policy_baseline_2x32 \
    --output-root outputs/cloud/smoke/leduc_dream_architecture_candidate_comparison" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"
```

Key outputs include `checkpoint_curves_by_variant.csv`,
`seed_variant_summary.csv`, `aggregate_summary_by_variant.csv`,
`paired_differences_vs_baseline.csv`, `paired_difference_summary.json`, and
standard node-axis plots under `plots/`, including
`dream_architecture_candidate_exploitability_by_nodes.png`.
