# DREAM final-only average-policy training ablation

This experiment tests whether the timing of average-policy network fitting affects measured DREAM performance on OpenSpiel Kuhn poker.

The advantage and baseline networks are still trained during traversal exactly as in the DREAM baseline. The ablation changes only the average-policy extraction schedule used for evaluation through the OpenSpiel policy interface:

- `intermittent_every_25_baseline` trains the average-policy network every 25 DREAM iterations and at the final iteration.
- `final_only_100_steps` trains the average-policy network only once at the end for one baseline policy-training event.
- `final_only_matched_steps` trains the average-policy network only once at the end with the same total policy-gradient budget as the intermittent arm.

Because traversal policies are generated from the advantage networks rather than from the average-policy network, this is best interpreted as a policy-extraction stability experiment rather than a test of changed regret-learning dynamics.

## Run

From the repository root:

```bash
python -m experiments.kuhn_poker.dream_final_only_policy_training_ablation.run
```

## Smoke test

```bash
python -m experiments.kuhn_poker.dream_final_only_policy_training_ablation.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --evaluation-interval 5 \
  --output-root outputs/smoke_tests/dream_final_only_policy_training_ablation
```

Key outputs include `checkpoint_curves_by_variant.csv`, `seed_variant_summary.csv`, `aggregate_summary_by_variant.csv`, `paired_differences_vs_intermediate_baseline.csv`, per-variant `seed_summary.json` files, and thesis-style plots under `plots/`.
