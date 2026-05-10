
# Output conventions

DREAM experiments should write outputs to a timestamped directory under `outputs/<experiment_name>/` unless a custom `--output-root` is supplied.

Each experiment should export:

- `experiment_metadata.json` — configuration and run metadata;
- `checkpoint_curves.csv` — per-seed, per-checkpoint metrics;
- `seed_summary.csv` — one summary row per seed and variant;
- `aggregate_summary.json` — aggregate mean and standard-error metrics;
- thesis-style `.png` plots with names that match the equivalent Deep CFR and ESCHER experiments where possible.

The core metric names should remain stable across experiments:

- `exploitability`
- `nash_conv`
- `policy_value_player_0`
- `policy_value_error`
- `nodes_touched`
- `wall_clock_seconds`
- `policy_loss`
- `advantage_target_variance`

This consistency makes it easier to combine Deep CFR, ESCHER, and DREAM results in the thesis.
