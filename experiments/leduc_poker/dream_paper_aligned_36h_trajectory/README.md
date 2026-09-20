# Paper-aligned DREAM 36-hour trajectory (Experiment 46)

## Research question

Does the selected paper-aligned DREAM configuration continue to improve beyond
the 12-hour horizon, and is that improvement stable across the five thesis
comparison seeds when measured against both active training time and nodes
touched?

## Frozen design

The source learner is exactly the selected Experiment 45 configuration:

- seeds `1234`, `2025`, `31415`, `27182`, and `16180`;
- one `n2-standard-8` VM per seed, with all five seeds running in parallel;
- 36 active training hours per seed;
- 900 traversals per player and iteration, `epsilon=0.50`, and linear weighting;
- 3,000 advantage updates with batch size 2,048 and a two-million-row
  reservoir per player;
- 1,000 baseline updates per iteration with batch size 512 and a 200,000-row
  circular buffer per player;
- freshly initialised `128x128` player advantage networks, `32x32` baseline and
  policy networks, Adam learning rate `0.001`, and gradient clipping at `1.0`;
- a shared four-million-row strategy reservoir; and
- no average-policy fitting or exploitability calculation inside the timed
  learner.

The training worker freezes iteration- and reach-weighted grouped sufficient
statistics every 30 minutes, at the first completed iteration crossing 15
million nodes, and at the final completed iteration after the 36-hour boundary.
Checkpoint I/O is excluded from active training time. Raw strategy reservoirs
are retained at 12 hours, the 15-million-node endpoint, and the final endpoint;
the other temporal checkpoints retain only the compact sufficient statistics.

Each checkpoint is then fitted from the same fresh seed-specific initialisation
using 4,000 updates of grouped, weighted soft-target cross-entropy. Evaluation
reports exact neural and empirical-policy exploitability, policy value,
distillation gap, adjacent-checkpoint exact head-to-head value, active time,
nodes touched, throughput, fitting cost, AUC, and final-window stability.

The primary endpoint is exact neural-policy exploitability at 36 hours. The
12-hour, 24-hour, 15-million-node, empirical-policy, and temporal summaries are
secondary evidence. Because three seeds are reused from Experiment 45, this is
an extended five-seed validation rather than a wholly fresh confirmation.

## Cloud execution

From the repository root, with `PROJECT_ID`, `REGION`, `BUCKET`, and `SA_EMAIL`
already set:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="drm46-paper36h-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=5

./gcp/run_dream_paper_aligned_36h_trajectory.sh smoke-local
./gcp/run_dream_paper_aligned_36h_trajectory.sh run
```

The remote controller performs a cloud smoke test, launches five parallel
training workers, launches five parallel deferred-evaluation workers, and then
aggregates the analysis. The laptop may be disconnected after submission.

Status and recovery use the same `RUN_ID`:

```bash
./gcp/run_dream_paper_aligned_36h_trajectory.sh status
./gcp/run_dream_paper_aligned_36h_trajectory.sh resume
```

The principal analysis artifacts are:

- `analysis/exploitability_by_training_time.png`;
- `analysis/exploitability_by_nodes.png`;
- `analysis/distillation_gap_by_training_time.png`;
- `analysis/policy_value_by_training_time.png`;
- `analysis/endpoint_aggregate_summary.csv`;
- `analysis/endpoint_seed_metrics.csv`;
- `analysis/trajectory_by_training_time_summary.csv`; and
- `analysis/trajectory_by_nodes_summary.csv`.
