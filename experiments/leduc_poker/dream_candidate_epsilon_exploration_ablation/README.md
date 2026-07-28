# DREAM candidate-architecture epsilon-exploration ablation (Experiment 23)

This experiment retests DREAM's epsilon-mixed traversal policy after adopting
the architecture-selected candidate as the new baseline. The network
configuration is fixed to:

- average-policy network: `[32, 32]`;
- player-specific advantage networks: `[128, 128]`;
- learned-baseline networks: `[32, 32]`.

The original epsilon ablation found that `epsilon=0.10` outperformed the old
`2x32` baseline value of `0.06`. This follow-up therefore retains `0.06`, reruns
`0.10`, and tests two larger values, `0.15` and `0.20`, to determine whether
the benefit of additional exploration continues under the new architecture.

The default run uses three matched seeds, `1234`, `2025`, and `31415`, matching
the original epsilon screen. All non-epsilon settings are held fixed: `175`
iterations, `160` traversals per player per iteration, exact exploitability
evaluation every `25` iterations, replay capacity `1e6`, learning rate `0.003`,
and intermittent average-policy training every `25` iterations.

The primary comparison metrics are final exploitability, final-window
exploitability, exploitability AUC by nodes touched, final policy-value error,
and paired seed-level deltas against `epsilon=0.06`.

## Run

```bash
python -m experiments.leduc_poker.dream_candidate_epsilon_exploration_ablation.run
```

Full runs reuse the tracked Experiment 22 candidate baseline comparator by default. Add `--train-baseline` only when you intentionally want to retrain that comparator.

## Smoke test

```bash
python -m experiments.leduc_poker.dream_candidate_epsilon_exploration_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants candidate_epsilon_006_baseline,candidate_epsilon_010 \
  --train-baseline \
  --output-root outputs/smoke_tests/dream_candidate_epsilon_exploration_ablation
```

## GCP Batch smoke test

```bash
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp23-candidate-epsilon-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_epsilon_exploration_ablation.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --variants candidate_epsilon_006_baseline,candidate_epsilon_010 \
    --train-baseline \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_epsilon_exploration_ablation" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"
```

Key outputs include `checkpoint_curves_by_variant.csv`,
`seed_variant_summary.csv`, `aggregate_summary_by_variant.csv`,
`paired_differences_vs_baseline.csv`, `paired_difference_summary.json`, and
standard node-axis plots under `plots/`, including
`dream_candidate_epsilon_exploration_exploitability_by_nodes.png`.
