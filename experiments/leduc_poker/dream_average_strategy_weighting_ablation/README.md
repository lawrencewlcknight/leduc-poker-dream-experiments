# DREAM average-strategy weighting ablation

This experiment tests whether the CFR-style iteration weighting used when
training the DREAM average-policy network improves Leduc poker performance. It
is the DREAM analogue of the Deep CFR replay/average-strategy weighting
ablation, but it preserves DREAM's sampled-traversal reach-ratio correction in
both arms.

The default variants are:

| Variant id | Average-policy loss multiplier |
| --- | --- |
| `linear_avg_importance_dream_baseline` | `sqrt(iteration * reach_ratio)`; current DREAM baseline. |
| `uniform_avg_importance` | `sqrt(reach_ratio)`; removes iteration weighting only. |

All other core settings are aligned with the DREAM baseline: OpenSpiel Leduc
poker, 175 DREAM iterations, 160 traversals per player per iteration, 32x32
policy/advantage/baseline networks, replay capacities, batch sizes, supervised
update budgets, exploration rate, policy-training cadence, and evaluation
interval.

## Run

From the repository root:

```bash
python -m experiments.leduc_poker.dream_average_strategy_weighting_ablation.run
```

Quick smoke test:

```bash
python -m experiments.leduc_poker.dream_average_strategy_weighting_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --batch-size-advantage 1 \
  --batch-size-strategy 1 \
  --batch-size-baseline 1 \
  --output-root outputs/smoke_tests/dream_average_strategy_weighting_ablation
```

Useful CLI options:

- `--variants` selects a comma-separated subset of weighting arms.
- `--allow-policy-training-rng-advance` disables RNG restoration around
  intermittent average-policy training.

## Outputs

The run directory follows `docs/OUTPUT_CONVENTIONS.md` and the existing DREAM
variant-ablation convention:

- `checkpoint_curves_by_variant.csv` - one row per checkpoint with weighting
  metadata, exploitability, value diagnostics, losses, replay sizes, entropy,
  and gradient norms.
- `seed_variant_summary.csv` - one row per `(variant, seed)`.
- `aggregate_summary_by_variant.csv` and `.json` - across-seed mean and
  standard-error summaries grouped by weighting variant.
- `paired_differences_vs_baseline.csv` - per-seed variant minus linear-weighted
  baseline deltas.
- `paired_difference_summary.json` - aggregate paired differences.
- `multiseed_curves_by_variant.npz` - compact curve arrays for downstream
  analysis.
- `plots/` - exploitability, average-policy value, policy-value error,
  final-metric, paired-delta, and policy-diagnostic plots.

Paired differences are reported as `variant - linear baseline`; negative
exploitability, AUC, or policy-value-error deltas are improvements.
