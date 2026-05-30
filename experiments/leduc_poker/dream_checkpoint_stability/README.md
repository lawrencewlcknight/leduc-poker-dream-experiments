# DREAM checkpoint-stability experiment

This experiment tests whether later DREAM average-policy checkpoints are stronger than earlier checkpoints in direct head-to-head play, and whether matchup strength agrees with exploitability.

It has two stages, matching the original notebook split:

- training saves lightweight average-policy snapshots at fixed checkpoint iterations;
- analysis loads those snapshots, computes exact pairwise seat-averaged expected values, and exports monotonicity and checkpoint-strength summaries.

The default command runs both stages:

```bash
python -m experiments.leduc_poker.dream_checkpoint_stability.run
```

To train snapshots without analysis:

```bash
python -m experiments.leduc_poker.dream_checkpoint_stability.run --stage train
```

To rerun analysis on an existing checkpoint run:

```bash
python -m experiments.leduc_poker.dream_checkpoint_stability.run \
  --stage analyze \
  --checkpoint-dir outputs/dream_checkpoint_stability/<timestamp>
```

## Smoke test

```bash
python -m experiments.leduc_poker.dream_checkpoint_stability.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --checkpoint-schedule 5,10 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --evaluation-interval 5 \
  --output-root outputs/smoke_tests/dream_checkpoint_stability
```

Key outputs include `checkpoint_curves.csv`, `seed_summary.csv`, `aggregate_summary.json`, policy snapshots under `policy_snapshots/`, and head-to-head analysis tables and plots under `head_to_head_analysis/`.
