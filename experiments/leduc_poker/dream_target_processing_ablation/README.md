# Leduc Poker DREAM Advantage-Target Processing Ablation

This experiment tests whether simple preprocessing of sampled DREAM advantage
targets improves optimisation stability and final average-policy quality in
Leduc poker. Replay buffers store raw sampled advantage targets for every
variant; processing is applied only to the supervised advantage-network fitting
loss.

The default variants are:

| Variant id | Processing |
| --- | --- |
| `raw_targets_dream_baseline` | Raw DREAM baseline targets. |
| `standardized_targets` | Batch-standardized advantage targets. |
| `clipped_targets` | Targets clipped to `[-1.0, 1.0]`. |
| `standardized_clipped_targets` | Batch-standardized, then clipped to `[-1.0, 1.0]`. |

All other core settings are aligned with the DREAM baseline: OpenSpiel Leduc
poker, 175 DREAM iterations, 160 traversals per player per iteration, 32x32
policy/advantage/baseline networks, replay capacities, batch sizes, supervised
update budgets, exploration rate, policy-training cadence, and evaluation
interval.

## Run

From the repository root:

```bash
python -m experiments.leduc_poker.dream_target_processing_ablation.run
```

Quick smoke test:

```bash
python -m experiments.leduc_poker.dream_target_processing_ablation.run \
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
  --variants raw_targets_dream_baseline,standardized_clipped_targets \
  --output-root outputs/smoke_tests/dream_target_processing_ablation
```

Useful CLI options:

- `--variants` selects a comma-separated subset of target-processing arms.
- `--target-clip-value` changes the clipping threshold for all variants.
- `--target-standardize-epsilon` changes the minimum scale used for
  standardisation.

## Outputs

The run directory follows `docs/OUTPUT_CONVENTIONS.md` and the existing DREAM
variant-ablation convention:

- `checkpoint_curves_by_variant.csv` — one row per checkpoint with
  target-processing metadata, exploitability, value diagnostics, raw target
  variance, processed target diagnostics, losses, replay sizes, entropy, and
  gradient norms.
- `seed_variant_summary.csv` — one row per `(variant, seed)`.
- `aggregate_summary_by_variant.csv` and `.json` — across-seed mean and
  standard-error summaries grouped by target-processing variant.
- `paired_differences_vs_baseline.csv` — per-seed variant minus raw-target
  baseline deltas.
- `paired_difference_summary.json` — aggregate paired differences.
- `multiseed_curves_by_variant.npz` — compact curve arrays for downstream
  analysis.
- `plots/` — exploitability, average-policy value, policy-value error,
  final-metric, paired-delta, and target-diagnostic plots.

Paired differences are reported as `variant - raw baseline`; negative
exploitability, AUC, or policy-value-error deltas are improvements.
