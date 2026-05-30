
# Output conventions

DREAM experiments should write outputs to a timestamped directory under `outputs/<experiment_name>/` unless a custom `--output-root` is supplied.

Each experiment should export:

- `experiment_metadata.json` — configuration and run metadata;
- `checkpoint_curves.csv` — per-seed, per-checkpoint metrics;
- `seed_summary.csv` — one summary row per seed and variant;
- `aggregate_summary.json` — aggregate mean and standard-error metrics;
- thesis-style `.png` plots with names that match the equivalent Deep CFR and ESCHER experiments where possible.

Variant or ablation experiments may use variant-specific filenames while preserving the same metric columns:

- `checkpoint_curves_by_variant.csv` — per-seed, per-variant, per-checkpoint metrics;
- `seed_variant_summary.csv` — one summary row per seed and variant;
- `aggregate_summary_by_variant.csv` and `.json` — aggregate mean and standard-error metrics by variant;
- `paired_differences_vs_<baseline>.csv` — matched-seed deltas against the configured baseline variant;
- `plots/*.png` — thesis-style figures with a stable experiment prefix.

Checkpoint-stability experiments should keep both stages under one timestamped run directory:

- `checkpoint_curves.csv` — per-seed checkpoint metrics from training;
- `policy_snapshots/*.pt` — lightweight average-policy snapshots for exact replay;
- `head_to_head_analysis/checkpoint_inventory.csv` — snapshots used by the analysis stage;
- `head_to_head_analysis/head_to_head_exact_pairwise.csv` — exact pairwise seat-averaged expected values;
- `head_to_head_analysis/head_to_head_exact_mean_matrix.csv` — mean checkpoint-vs-checkpoint EV matrix;
- `head_to_head_analysis/head_to_head_seed_win_fraction_matrix.csv` — seed win-fraction matrix;
- `head_to_head_analysis/head_to_head_monotonicity_summary_by_seed.csv` — monotonicity rates by seed;
- `head_to_head_analysis/head_to_head_aggregate_strength_summary.csv` — aggregate strength and exploitability by checkpoint;
- `head_to_head_analysis/best_checkpoint_summary.csv` — best checkpoint by head-to-head EV and by exploitability.

Random-search experiments should keep screening and confirmation outputs under one timestamped run directory:

- `screening_configs.json` — baseline plus sampled screening candidates;
- `confirmation_configs.json` — baseline plus selected candidates promoted from screening;
- `tables/<stage>_run_summaries.csv` — one row per config/seed run;
- `tables/<stage>_curves.csv` — per-checkpoint curves for all config/seed runs;
- `tables/<stage>_config_summary.csv` — aggregate mean and standard-error metrics by configuration;
- `tables/confirmation_paired_differences_final_exploitability.csv` — matched-seed deltas against the baseline;
- `traces/<stage>/<config_label>/seed_<seed>_curves.csv` — per-run trace files;
- `traces/<stage>/<config_label>/seed_<seed>_summary.json` — per-run summaries;
- `plots/*.png` — thesis-style screening ranks, confirmation bars, paired deltas, and curves.

Warm-start ablations should keep both paired arms under one timestamped run directory:

- `checkpoint_curves_by_arm.csv` — per-seed, per-arm checkpoint metrics;
- `seed_arm_summary.csv` — one summary row per seed and arm;
- `aggregate_summary_by_arm.csv` and `.json` — aggregate mean and standard-error metrics by arm;
- `paired_differences_warm_minus_baseline.csv` — matched-seed warm-start minus continuous deltas;
- `paired_difference_summary.json` — aggregate paired-difference means and standard errors;
- `seed_<seed>/<arm>/checkpoint_curves.csv` — per-seed arm traces;
- `seed_<seed>/warm_start_resume/checkpoint/*.pt` — saved full solver states used for resume;
- `plots/*.png` — thesis-style arm curves, summary bars, and paired-difference plots.

Learning-rate schedule ablations should keep all schedule variants under one timestamped run directory:

- `checkpoint_curves_by_variant.csv` — per-seed, per-schedule checkpoint metrics;
- `seed_variant_summary.csv` — one summary row per seed and schedule variant;
- `aggregate_summary_by_variant.csv` and `.json` — aggregate mean and standard-error metrics by schedule variant;
- `paired_differences_vs_baseline.csv` — matched-seed schedule deltas against the constant-learning-rate DREAM baseline;
- `paired_difference_summary.json` — aggregate paired-difference means and standard errors;
- `multiseed_curves_by_variant.npz` — array export for thesis plotting or notebook reuse;
- `seed_<seed>/<variant>/checkpoint_curves.csv` — per-run trace files;
- `plots/*.png` — thesis-style schedule curves, summary bars, paired-difference plots, and the learning-rate schedule trace.

Network-training budget ablations should keep all budget variants under one timestamped run directory:

- `checkpoint_curves_by_variant.csv` — per-seed, per-budget checkpoint metrics;
- `seed_variant_summary.csv` — one summary row per seed and budget variant;
- `aggregate_summary_by_variant.csv` and `.json` — aggregate mean and standard-error metrics by budget variant;
- `paired_differences_vs_baseline.csv` — matched-seed budget deltas against the configured baseline budget;
- `paired_difference_summary.json` — aggregate paired-difference means and standard errors;
- `multiseed_curves_by_variant.npz` — array export for thesis plotting or notebook reuse;
- `seed_<seed>/<variant>/checkpoint_curves.csv` — per-run trace files;
- `plots/*.png` — thesis-style budget curves, summary bars, diagnostic plots, and paired-difference plots.

Scalar-parameter ablations, such as epsilon-exploration sweeps, should keep all parameter variants under one timestamped run directory:

- `checkpoint_curves_by_variant.csv` — per-seed, per-variant checkpoint metrics;
- `seed_variant_summary.csv` — one summary row per seed and variant;
- `aggregate_summary_by_variant.csv` and `.json` — aggregate mean and standard-error metrics by variant;
- `paired_differences_vs_baseline.csv` — matched-seed parameter deltas against the configured baseline variant;
- `paired_difference_summary.json` — aggregate paired-difference means and standard errors;
- `multiseed_curves_by_variant.npz` — array export for thesis plotting or notebook reuse;
- `seed_<seed>/<variant>/checkpoint_curves.csv` — per-run trace files;
- `plots/*.png` — thesis-style parameter curves, summary bars, diagnostic plots, and paired-difference plots.

Trajectory-count ablations should additionally expose sample-efficiency columns and plots:

- `sampled_trajectories` — cumulative outcome-sampling trajectories across both players;
- `trajectories_per_iteration` — per-iteration total outcome-sampling trajectories across both players;
- `exploitability_auc_by_sampled_trajectories` — normalised AUC over sampled trajectories;
- `sampled_trajectories_to_threshold` — first sampled-trajectory count at which the exploitability threshold is reached;
- `plots/*_exploitability_by_sampled_trajectories.png` — cost-normalised exploitability curve;
- `plots/*_paired_sample_trajectory_auc_delta.png` — matched-seed sample-efficiency delta plot.

The core metric names should remain stable across experiments:

- `exploitability`
- `nash_conv`
- `policy_value_player_0`
- `average_policy_value`
- `average_policy_value_target`
- `average_policy_value_error`
- `policy_value_error`
- `nodes_touched`
- `wall_clock_seconds`
- `policy_loss`
- `policy_training_events`
- `policy_gradient_steps_total`
- `advantage_target_variance`
- `learning_rate`
- `baseline_network_train_steps`
- `baseline_loss_mean`
- `baseline_reward_variance_sampled`
- `epsilon`

`policy_value_player_0` is the raw OpenSpiel player-0 value. `average_policy_value` uses the same player-0 convention for headline charts. In Leduc poker the default plotted target is approximately `-0.085606`. The target is configured per experiment through `average_policy_value_target` so the same plotting code can be reused for games with a different equilibrium value.
- `num_traversals`
- `sampled_trajectories`
- `trajectories_per_iteration`

Random-search summary columns should also remain stable:

- `config_label`
- `stage`
- `status`
- `final_exploitability_mean`
- `final_window_mean_exploitability_mean`
- `best_exploitability_mean`
- `exploitability_auc_by_iteration_mean`
- `final_policy_value_error_mean`
- `train_seconds_mean`

Warm-start paired-difference columns should use the suffix:

- `_warm_minus_baseline`

Learning-rate schedule variant columns should remain stable:

- `variant`
- `learning_rate_schedule`
- `learning_rate_start`
- `learning_rate_end`
- `final_learning_rate`
- `delta_final_exploitability`
- `delta_final_window_mean_exploitability`
- `delta_best_exploitability`

Network-budget variant columns should remain stable:

- `variant`
- `variant_label`
- `baseline_network_train_steps`
- `final_baseline_loss_mean`
- `final_baseline_reward_variance`
- `final_advantage_target_variance`
- `delta_final_exploitability`
- `delta_final_baseline_loss_mean`
- `delta_final_advantage_target_variance`

Epsilon-exploration variant columns should remain stable:

- `variant`
- `variant_label`
- `epsilon`
- `final_advantage_target_variance`
- `final_baseline_reward_variance`
- `final_policy_entropy_mean`
- `delta_final_exploitability`
- `delta_final_policy_value_error`
- `delta_final_advantage_target_variance`

Trajectories-per-iteration variant columns should remain stable:

- `variant`
- `variant_label`
- `num_traversals`
- `trajectories_per_iteration`
- `final_sampled_trajectories`
- `exploitability_auc_by_sampled_trajectories`
- `sampled_trajectories_to_threshold`
- `delta_final_exploitability`
- `delta_exploitability_auc_by_nodes`
- `delta_exploitability_auc_by_sampled_trajectories`

Checkpoint head-to-head metric names should also remain stable:

- `checkpoint`
- `checkpoint_A`
- `checkpoint_B`
- `A_EV_as_player_0`
- `A_EV_as_player_1`
- `A_EV_seat_averaged`
- `mean_EV_vs_all_other_checkpoints`
- `mean_EV_vs_earlier`
- `EV_vs_previous`

This consistency makes it easier to combine Deep CFR, ESCHER, and DREAM results in the thesis.
