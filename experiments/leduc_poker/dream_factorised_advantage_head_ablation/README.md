# DREAM factorised advantage-head ablation

This experiment tests whether structuring the DREAM advantage-network output
head improves optimisation stability or final average-policy quality in Leduc
poker. It is the DREAM analogue of the Deep CFR factorised advantage-head
ablation.

The policy network and learned baseline/control-variate networks stay fixed as
plain `2x32` MLPs. The intended treatment variable is only the player-specific
advantage-network output head:

| Treatment | Network type | Description |
| --- | --- | --- |
| `direct` | `mlp` | Baseline action-value outputs. |
| `centered` | `centered_advantage_mlp` | Action outputs centred to zero mean per information state. |
| `dueling` | `dueling_mlp` | Scalar state-value head plus centred action-advantage outputs. |

Each treatment is run at hidden depths `2`, `4`, and `8`, all at width `32`.
All non-architecture settings match the aligned DREAM architecture-ablation
baseline.

## Run

```bash
python -m experiments.leduc_poker.dream_factorised_advantage_head_ablation.run
```

## Smoke test

```bash
python -m experiments.leduc_poker.dream_factorised_advantage_head_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --variants direct_advantage_layers2_width32,centered_advantage_layers2_width32,dueling_advantage_layers2_width32 \
  --output-root outputs/smoke_tests/dream_factorised_advantage_head_ablation
```

Key outputs follow the DREAM architecture-ablation convention:
`checkpoint_curves_by_variant.csv`, `seed_variant_summary.csv`,
`aggregate_summary_by_variant.csv`, `paired_differences_vs_baseline.csv`,
`paired_difference_summary.json`, `multiseed_curves_by_variant.npz`, and plots
under `plots/`.
