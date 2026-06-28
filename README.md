
# Leduc Poker DREAM Experiments

This repository contains reproducible experiments for evaluating a DREAM-style neural regret-minimisation algorithm on Leduc poker using DeepMind's OpenSpiel library.

The immediate aim is to establish a thesis-quality DREAM baseline that is aligned with the sister Deep CFR and ESCHER repositories. Leduc poker is used as the diagnostic environment because it is a small two-player zero-sum imperfect-information game with a known game value and exact exploitability evaluation. The results from this repository are intended to sit alongside the Deep CFR and ESCHER Leduc poker experiments in an MPhil thesis on neural CFR methods for poker.

The repository is organised so that each experiment can be run independently while sharing reusable DREAM code. The shared `dream_poker` package contains the DREAM-style solver, neural-network definitions, replay buffers, plotting helpers, seeding utilities, and experiment export utilities. Each experiment lives in its own package under `experiments/leduc_poker/<experiment_name>/`.

> Important note: this is an OpenSpiel DREAM-style implementation designed for comparable thesis experiments. It is not a bit-for-bit port of the official PokerRL DREAM repository.

## Repository structure

```text
.
├── dream_poker/                                      # Shared reusable code
│   ├── solver.py                                     # DREAM-style OpenSpiel solver
│   ├── networks.py                                   # MLP networks for policy, advantage, baseline
│   ├── replay.py                                     # Reservoir and circular replay buffers
│   ├── checkpointing.py                              # Policy snapshots and head-to-head analysis
│   ├── experiment_utils.py                           # Run-dir, metric, and export helpers
│   ├── experiment_runner.py                          # Shared experiment-runner helpers
│   ├── random_search.py                              # Staged solver-parameter search helpers
│   ├── warm_start.py                                 # Warm-start paired-analysis helpers
│   ├── lr_schedule.py                                # Learning-rate schedule ablation helpers
│   ├── network_budget.py                             # Network-update budget ablation helpers
│   ├── variant_ablation.py                           # Generic matched-variant ablation helpers
│   ├── plotting.py                                   # Thesis-style plots
│   ├── constants.py                                  # Leduc value, thresholds, seed lists
│   └── seeding.py                                    # PyTorch/NumPy/Python seeding helpers
├── experiments/
│   └── leduc_poker/
│       ├── dream_multiseed_baseline/                 # Experiment 1
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_final_only_policy_training_ablation/ # Experiment 2
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_checkpoint_stability/               # Experiment 3
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_constrained_random_search/          # Experiment 4
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_warm_start_ablation/                # Experiment 5
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_lr_schedule_ablation/               # Experiment 6
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_baseline_network_budget_ablation/   # Experiment 7
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_epsilon_exploration_ablation/       # Experiment 8
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_trajectories_per_iteration_ablation/ # Experiment 9
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_network_size_ablation/               # Experiment 10: width
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_network_depth_ablation/              # Experiment 11
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_network_capacity_extremes_ablation/  # Experiment 12
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_target_processing_ablation/           # Experiment 13
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_residual_network_ablation/            # Experiment 14
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_average_strategy_weighting_ablation/  # Experiment 15
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       └── dream_factorised_advantage_head_ablation/   # Experiment 16
│           ├── config.py
│           ├── run.py
│           └── README.md
├── docs/
│   └── OUTPUT_CONVENTIONS.md
├── notebooks/                                       # Original notebook archive
├── outputs/                                         # Experiment outputs (gitignored)
├── scripts/                                         # Utility scripts
├── thesis_artifacts/                                # Curated thesis-facing artifacts
├── tests/                                           # Lightweight import/unit tests
├── venv/                                            # Placeholder only; environment not committed
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── README.md
└── TESTING.md
```

## Experiments

### 1. Leduc poker DREAM-style multi-seed baseline

[`experiments/leduc_poker/dream_multiseed_baseline/`](experiments/leduc_poker/dream_multiseed_baseline/README.md)

Runs the aligned DREAM-style baseline on OpenSpiel `leduc_poker` across matched random seeds. The primary metric is exploitability, reported as NashConv divided by two. Secondary metrics include policy-value error from the known Leduc game value, nodes touched, wall-clock time, and final/best/final-window exploitability. Diagnostic metrics include policy loss, advantage-network loss, learned-baseline loss, replay-buffer sizes, target variance, policy entropy, and gradient norms.

**Question:** under a fixed training protocol, does the DREAM-style implementation learn a low-exploitability average policy in Leduc poker, and how variable is the result across random seeds?

### 2. DREAM final-only average-policy training ablation

[`experiments/leduc_poker/dream_final_only_policy_training_ablation/`](experiments/leduc_poker/dream_final_only_policy_training_ablation/README.md)

Runs the DREAM-style baseline with three average-policy extraction schedules: intermittent training every 25 iterations, final-only training for one policy-update event, and final-only training with the same total policy-gradient budget as the intermittent arm. The advantage and learned-baseline updates are held fixed.

**Question:** does training the average-policy network intermittently during DREAM training affect measured exploitability, or is final-only policy extraction stable enough for evaluation and play through the OpenSpiel policy interface?

### 3. DREAM checkpoint-stability head-to-head analysis

[`experiments/leduc_poker/dream_checkpoint_stability/`](experiments/leduc_poker/dream_checkpoint_stability/README.md)

Runs a two-stage checkpoint-stability experiment. The training stage saves lightweight average-policy network snapshots at fixed iterations. The analysis stage reloads those snapshots and computes exact pairwise, seat-averaged head-to-head expected values between checkpoints, alongside exploitability and monotonicity summaries.

**Question:** do later DREAM average-policy checkpoints reliably beat earlier checkpoints in direct play, and does direct matchup strength agree with exploitability?

### 4. DREAM constrained solver-parameter random search

[`experiments/leduc_poker/dream_constrained_random_search/`](experiments/leduc_poker/dream_constrained_random_search/README.md)

Runs a bounded two-stage random search over DREAM solver parameters, including traversal count, exploration rate, network width/depth, learning rate, replay-memory capacity, batch size, and supervised update budgets. Screening identifies promising configurations; confirmation reruns the baseline and selected candidates with a longer budget and matched seeds.

**Question:** can DREAM performance in Leduc poker be improved materially by tuning implementation and optimisation parameters while keeping the DREAM-style algorithm fixed?

### 5. DREAM fair warm-start ablation

[`experiments/leduc_poker/dream_warm_start_ablation/`](experiments/leduc_poker/dream_warm_start_ablation/README.md)

Runs a paired continuous-vs-checkpoint/resume comparison. The warm-start arm saves the full DREAM solver state at an intermediate iteration, reloads it into a fresh solver, and resumes to the same final training budget as the continuous arm.

**Question:** does checkpointing and resuming a DREAM run behave comparably to an otherwise identical continuous run, without extra data or optimisation?

### 6. DREAM learning-rate schedule ablation

[`experiments/leduc_poker/dream_lr_schedule_ablation/`](experiments/leduc_poker/dream_lr_schedule_ablation/README.md)

Runs a paired constant-vs-decayed learning-rate comparison for the DREAM-style baseline. The OpenSpiel game, total training budget, traversal count, replay capacities, network architectures, supervised update budgets, average-policy training schedule, exploration rate, and matched seeds are held fixed; the intended treatment variable is only the optimiser learning-rate schedule.

**Question:** does a decaying learning-rate schedule improve DREAM performance in Leduc poker relative to the aligned constant-learning-rate baseline?

### 7. DREAM baseline-network training-budget ablation

[`experiments/leduc_poker/dream_baseline_network_budget_ablation/`](experiments/leduc_poker/dream_baseline_network_budget_ablation/README.md)

Runs a matched-seed ablation over the number of supervised updates allocated to the learned baseline/control-variate networks. The DREAM baseline arm uses 50 baseline-network updates per player per iteration; comparison arms use 25 and 100 updates while holding traversal budget, advantage-network training, average-policy training, architecture, replay capacity, learning rate, exploration, and seeds fixed.

**Question:** is DREAM performance in Leduc poker limited by baseline-network fitting budget, as measured by exploitability, policy-value error, baseline diagnostics, and advantage-target variance?

### 8. DREAM epsilon-exploration ablation

[`experiments/leduc_poker/dream_epsilon_exploration_ablation/`](experiments/leduc_poker/dream_epsilon_exploration_ablation/README.md)

Runs a matched-seed ablation over the epsilon-mixed sampling policy used during DREAM outcome-sampling traversals. The DREAM baseline arm uses epsilon `0.06`; comparison arms use `0.03` and `0.10` while holding traversal budget, network training budgets, architecture, replay capacity, learning rate, average-policy training schedule, and seeds fixed.

**Question:** how does the exploration rate affect game-tree coverage, target variance, policy extraction, and final exploitability in Leduc poker?

### 9. DREAM trajectories-per-iteration ablation

[`experiments/leduc_poker/dream_trajectories_per_iteration_ablation/`](experiments/leduc_poker/dream_trajectories_per_iteration_ablation/README.md)

Runs a matched-seed ablation over the number of outcome-sampling traversals per player per DREAM iteration. The DREAM baseline arm uses 160 traversals; comparison arms use 80 and 320 traversals while holding the training schedule, network budgets, architecture, replay capacity, learning rate, exploration rate, and seeds fixed.

**Question:** does increasing trajectories per iteration improve DREAM performance in Leduc poker, and does any gain remain when performance is measured by nodes touched or sampled trajectories rather than iteration count?

### 10. DREAM network-width ablation

[`experiments/leduc_poker/dream_network_size_ablation/`](experiments/leduc_poker/dream_network_size_ablation/README.md)

Runs a matched-seed width sweep over `[16, 16]`, baseline `[32, 32]`, `[64, 64]`, and `[128, 128]` while holding network depth and all non-architecture settings fixed.

**Question:** how does hidden-layer width affect DREAM exploitability, policy-value error, sample efficiency, and network diagnostics in Leduc poker?

### 11. DREAM network-depth ablation

[`experiments/leduc_poker/dream_network_depth_ablation/`](experiments/leduc_poker/dream_network_depth_ablation/README.md)

Runs a matched-seed depth sweep over `[32]`, baseline `[32, 32]`, and `[32, 32, 32]` while holding hidden width and all non-architecture settings fixed.

**Question:** how does hidden-layer depth affect DREAM performance at a fixed width of 32 units?

### 12. DREAM network-capacity extremes ablation

[`experiments/leduc_poker/dream_network_capacity_extremes_ablation/`](experiments/leduc_poker/dream_network_capacity_extremes_ablation/README.md)

Runs a matched-seed comparison of low-capacity `[16]`, baseline `[32, 32]`, and high-capacity `[64, 64, 64]` networks.

**Question:** does DREAM benefit from substantially more representational capacity, or does the Leduc poker problem favour compact networks?

### 13. DREAM advantage-target processing ablation

[`experiments/leduc_poker/dream_target_processing_ablation/`](experiments/leduc_poker/dream_target_processing_ablation/README.md)

Runs a matched-seed comparison of raw DREAM advantage targets, batch-standardized targets, clipped targets, and standardized-then-clipped targets. Replay buffers store raw sampled targets for every arm; processing is applied only to the supervised advantage-network loss.

**Question:** can simple advantage-target processing reduce DREAM optimisation instability and improve final average-policy quality in Leduc poker?

### 14. DREAM residual-network ablation

[`experiments/leduc_poker/dream_residual_network_ablation/`](experiments/leduc_poker/dream_residual_network_ablation/README.md)

Compares plain MLPs against residual MLPs at fixed width `32` and hidden depths `2`, `4`, and `8`. The same treatment is applied to the average-policy, advantage, and learned-baseline networks within each variant.

**Question:** do residual skip connections improve DREAM optimisation stability or final average-policy quality when all non-architecture settings are held fixed?

### 15. DREAM average-strategy weighting ablation

[`experiments/leduc_poker/dream_average_strategy_weighting_ablation/`](experiments/leduc_poker/dream_average_strategy_weighting_ablation/README.md)

Compares the current DREAM average-policy target weighting, `sqrt(iteration * reach_ratio)`, with uniform average-strategy iteration weighting, `sqrt(reach_ratio)`. Both arms retain DREAM's sampled-traversal reach-ratio correction; the intended treatment variable is only the CFR-style iteration weighting used in the average-policy supervised loss.

**Question:** does average-strategy iteration weighting improve DREAM stability or final average-policy quality when every other core training parameter is held fixed?

### 16. DREAM factorised advantage-head ablation

[`experiments/leduc_poker/dream_factorised_advantage_head_ablation/`](experiments/leduc_poker/dream_factorised_advantage_head_ablation/README.md)

Holds the average-policy and learned-baseline networks fixed at the baseline `2x32` MLP architecture and compares direct action outputs with centred action-advantage outputs and dueling-style state-value-plus-action-advantage outputs for the player-specific advantage networks. The comparison is run at hidden depths `2`, `4`, and `8`, all at width `32`.

**Question:** does imposing a value/advantage factorisation on the DREAM advantage approximator improve optimisation stability or final average-policy quality?

Future DREAM ablations should be added as separate experiment folders under `experiments/leduc_poker/`, while reusing the shared `dream_poker` package and output conventions.

## Setup

Create and activate a virtual environment. The repository contains a placeholder `venv/` directory, but the actual environment is not committed.

```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
# .\venv\Scripts\Activate.ps1   # Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```

OpenSpiel installation can vary by platform. If `pip install -r requirements.txt` fails on `open_spiel`, install OpenSpiel following the official instructions for your platform.

## Running the experiments

From the repository root:

```bash
# Experiment 1 — full aligned DREAM-style baseline
python -m experiments.leduc_poker.dream_multiseed_baseline.run

# Experiment 2 — final-only average-policy training ablation
python -m experiments.leduc_poker.dream_final_only_policy_training_ablation.run

# Experiment 3 — checkpoint-stability training plus head-to-head analysis
python -m experiments.leduc_poker.dream_checkpoint_stability.run

# Experiment 4 — constrained solver-parameter random search
python -m experiments.leduc_poker.dream_constrained_random_search.run

# Experiment 5 — fair warm-start/checkpoint-resume ablation
python -m experiments.leduc_poker.dream_warm_start_ablation.run

# Experiment 6 — learning-rate schedule ablation
python -m experiments.leduc_poker.dream_lr_schedule_ablation.run

# Experiment 7 — baseline-network training-budget ablation
python -m experiments.leduc_poker.dream_baseline_network_budget_ablation.run

# Experiment 8 — epsilon-exploration ablation
python -m experiments.leduc_poker.dream_epsilon_exploration_ablation.run

# Experiment 9 — trajectories-per-iteration ablation
python -m experiments.leduc_poker.dream_trajectories_per_iteration_ablation.run

# Experiment 10 — network-width ablation
python -m experiments.leduc_poker.dream_network_size_ablation.run

# Experiment 11 — network-depth ablation
python -m experiments.leduc_poker.dream_network_depth_ablation.run

# Experiment 12 — network-capacity extremes ablation
python -m experiments.leduc_poker.dream_network_capacity_extremes_ablation.run

# Experiment 13 — advantage-target processing ablation
python -m experiments.leduc_poker.dream_target_processing_ablation.run

# Experiment 14 — residual-network ablation
python -m experiments.leduc_poker.dream_residual_network_ablation.run

# Experiment 15 — average-strategy weighting ablation
python -m experiments.leduc_poker.dream_average_strategy_weighting_ablation.run

# Experiment 16 — factorised advantage-head ablation
python -m experiments.leduc_poker.dream_factorised_advantage_head_ablation.run
```

To run quick smoke tests for later DREAM ablations on GCP, use the Batch
submission script. These commands do not require the repository Python
dependencies to be installed locally; they only require the Google Cloud CLI,
the GCP environment variables from
[`docs/GCP_BATCH_EXPERIMENTS.md`](docs/GCP_BATCH_EXPERIMENTS.md), and a local
`python3` capable of running the submission helper.

```bash
# Submit Experiments 13-16 together with one shared timestamp.
./gcp/submit_recent_ablation_smoke_tests.sh

# Leduc Experiment 10 — network-width ablation smoke test on GCP
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp10-width-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_network_size_ablation.run \
    --seeds 1234 \
    --iterations 10 \
    --traversals 50 \
    --policy-network-train-steps 20 \
    --advantage-network-train-steps 20 \
    --baseline-network-train-steps 20 \
    --evaluation-interval 5 \
    --output-root outputs/cloud/smoke/leduc_dream_network_width_ablation" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"

# Leduc Experiment 11 — network-depth ablation smoke test on GCP
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp11-depth-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_network_depth_ablation.run \
    --seeds 1234 \
    --iterations 10 \
    --traversals 50 \
    --policy-network-train-steps 20 \
    --advantage-network-train-steps 20 \
    --baseline-network-train-steps 20 \
    --evaluation-interval 5 \
    --output-root outputs/cloud/smoke/leduc_dream_network_depth_ablation" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"

# Leduc Experiment 12 — network-capacity extremes ablation smoke test on GCP
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp12-capacity-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_network_capacity_extremes_ablation.run \
    --seeds 1234 \
    --iterations 10 \
    --traversals 50 \
    --policy-network-train-steps 20 \
    --advantage-network-train-steps 20 \
    --baseline-network-train-steps 20 \
    --evaluation-interval 5 \
    --output-root outputs/cloud/smoke/leduc_dream_network_capacity_extremes_ablation" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"

# Leduc Experiment 13 — advantage-target processing ablation smoke test on GCP
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp13-target-processing-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_target_processing_ablation.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --batch-size-advantage 1 \
    --batch-size-strategy 1 \
    --batch-size-baseline 1 \
    --evaluation-interval 1 \
    --variants raw_targets_dream_baseline,standardized_clipped_targets \
    --output-root outputs/cloud/smoke/leduc_dream_target_processing_ablation" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"

# Leduc Experiment 14 — residual-network ablation smoke test on GCP
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp14-residual-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_residual_network_ablation.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --variants plain_layers2_width32,residual_layers2_width32 \
    --output-root outputs/cloud/smoke/leduc_dream_residual_network_ablation" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"

# Leduc Experiment 15 — average-strategy weighting ablation smoke test on GCP
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp15-avg-weighting-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_average_strategy_weighting_ablation.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --batch-size-advantage 1 \
    --batch-size-strategy 1 \
    --batch-size-baseline 1 \
    --evaluation-interval 1 \
    --output-root outputs/cloud/smoke/leduc_dream_average_strategy_weighting_ablation" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"

# Leduc Experiment 16 — factorised advantage-head ablation smoke test on GCP
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp16-factorised-head-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_factorised_advantage_head_ablation.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --variants direct_advantage_layers2_width32,centered_advantage_layers2_width32,dueling_advantage_layers2_width32 \
    --output-root outputs/cloud/smoke/leduc_dream_factorised_advantage_head_ablation" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"
```

For a quick local smoke test of later DREAM ablations:

Run these from an activated environment that has `requirements.txt` installed. If
you see `ModuleNotFoundError: No module named 'matplotlib'`, the selected
`python` is not the environment used for this repository.

```bash
# Leduc Experiment 10 — network-width ablation smoke test
python -m experiments.leduc_poker.dream_network_size_ablation.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --evaluation-interval 5 \
  --output-root outputs/smoke_tests/dream_network_width_ablation

# Leduc Experiment 11 — network-depth ablation smoke test
python -m experiments.leduc_poker.dream_network_depth_ablation.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --evaluation-interval 5 \
  --output-root outputs/smoke_tests/dream_network_depth_ablation

# Leduc Experiment 12 — network-capacity extremes ablation smoke test
python -m experiments.leduc_poker.dream_network_capacity_extremes_ablation.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --evaluation-interval 5 \
  --output-root outputs/smoke_tests/dream_network_capacity_extremes_ablation

# Leduc Experiment 14 — residual-network ablation smoke test
python -m experiments.leduc_poker.dream_residual_network_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants plain_layers2_width32,residual_layers2_width32 \
  --output-root outputs/smoke_tests/dream_residual_network_ablation

python -m experiments.leduc_poker.dream_average_strategy_weighting_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --batch-size-advantage 1 \
  --batch-size-strategy 1 \
  --batch-size-baseline 1 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests/dream_average_strategy_weighting_ablation

python -m experiments.leduc_poker.dream_factorised_advantage_head_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants direct_advantage_layers2_width32,centered_advantage_layers2_width32,dueling_advantage_layers2_width32 \
  --output-root outputs/smoke_tests/dream_factorised_advantage_head_ablation
```

For local smoke tests across all experiments:

```bash
python -m experiments.leduc_poker.dream_multiseed_baseline.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --evaluation-interval 5 \
  --output-root outputs/smoke_tests/dream_multiseed_baseline

python -m experiments.leduc_poker.dream_final_only_policy_training_ablation.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --evaluation-interval 5 \
  --output-root outputs/smoke_tests/dream_final_only_policy_training_ablation

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

python -m experiments.leduc_poker.dream_constrained_random_search.run \
  --screening-seeds 1234 \
  --confirmation-seeds 1234 \
  --screening-iterations 5 \
  --confirmation-iterations 5 \
  --n-random-candidates 1 \
  --n-confirmation-candidates 1 \
  --num-traversals 20 \
  --policy-network-train-steps 5 \
  --advantage-network-train-steps 5 \
  --baseline-network-train-steps 5 \
  --evaluation-interval 5 \
  --output-root outputs/smoke_tests/dream_constrained_random_search

python -m experiments.leduc_poker.dream_warm_start_ablation.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --warm-start-iteration 5 \
  --traversals 50 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --evaluation-interval 5 \
  --output-root outputs/smoke_tests/dream_warm_start_ablation

python -m experiments.leduc_poker.dream_lr_schedule_ablation.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --evaluation-interval 5 \
  --output-root outputs/smoke_tests/dream_lr_schedule_ablation

python -m experiments.leduc_poker.dream_baseline_network_budget_ablation.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --evaluation-interval 5 \
  --variants baseline_steps_25,baseline_steps_50_exp_baseline \
  --output-root outputs/smoke_tests/dream_baseline_network_budget_ablation

python -m experiments.leduc_poker.dream_epsilon_exploration_ablation.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --evaluation-interval 5 \
  --output-root outputs/smoke_tests/dream_epsilon_exploration_ablation

python -m experiments.leduc_poker.dream_trajectories_per_iteration_ablation.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --evaluation-interval 5 \
  --variants traversals_80,traversals_160_exp_baseline \
  --output-root outputs/smoke_tests/dream_trajectories_per_iteration_ablation

python -m experiments.leduc_poker.dream_target_processing_ablation.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --batch-size-advantage 1 \
  --batch-size-strategy 1 \
  --batch-size-baseline 1 \
  --evaluation-interval 5 \
  --variants raw_targets_dream_baseline,standardized_clipped_targets \
  --output-root outputs/smoke_tests/dream_target_processing_ablation

python -m experiments.leduc_poker.dream_residual_network_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants plain_layers2_width32,residual_layers2_width32 \
  --output-root outputs/smoke_tests/dream_residual_network_ablation

python -m experiments.leduc_poker.dream_average_strategy_weighting_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --batch-size-advantage 1 \
  --batch-size-strategy 1 \
  --batch-size-baseline 1 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests/dream_average_strategy_weighting_ablation

python -m experiments.leduc_poker.dream_factorised_advantage_head_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants direct_advantage_layers2_width32,centered_advantage_layers2_width32,dueling_advantage_layers2_width32 \
  --output-root outputs/smoke_tests/dream_factorised_advantage_head_ablation
```

Outputs are written to a timestamped subdirectory under `outputs/` by default. Treat full `outputs/` directories as scratch data; promote only curated, lightweight thesis-facing artifacts into `thesis_artifacts/` using the workflow in [`docs/THESIS_ARTIFACTS.md`](docs/THESIS_ARTIFACTS.md).

The key files are:

```text
seed_summary.csv
aggregate_summary.json
checkpoint_curves.csv
experiment_metadata.json
exploitability_by_iteration_multiseed.png
exploitability_by_nodes_multiseed.png
average_policy_value_by_iteration_multiseed.png
average_policy_value_by_nodes_multiseed.png
policy_value_error_multiseed.png
policy_loss_diagnostic.png
advantage_target_variance_diagnostic.png
baseline_reward_variance_diagnostic.png
summary_metrics.png
```

Variant ablations additionally export:

```text
checkpoint_curves_by_variant.csv
seed_variant_summary.csv
aggregate_summary_by_variant.csv
aggregate_summary_by_variant.json
paired_differences_vs_intermediate_baseline.csv
plots/dream_policy_training_final_exploitability.png
plots/dream_policy_training_final_average_policy_value.png
plots/dream_policy_training_average_policy_value_by_iteration.png
plots/dream_policy_training_paired_delta_exploitability.png
```

Checkpoint-stability experiments additionally export:

```text
policy_snapshots/leduc_poker_dream_seed_<seed>_policy_snapshot_<iteration>_iters.pt
head_to_head_analysis/checkpoint_inventory.csv
head_to_head_analysis/checkpoint_exploitability_metrics.csv
head_to_head_analysis/head_to_head_exact_pairwise.csv
head_to_head_analysis/head_to_head_exact_mean_matrix.csv
head_to_head_analysis/head_to_head_seed_win_fraction_matrix.csv
head_to_head_analysis/head_to_head_monotonicity_summary_by_seed.csv
head_to_head_analysis/head_to_head_strength_with_metrics.csv
head_to_head_analysis/head_to_head_aggregate_strength_summary.csv
head_to_head_analysis/best_checkpoint_summary.csv
head_to_head_analysis/plots/dream_head_to_head_exact_mean_matrix.png
head_to_head_analysis/plots/dream_checkpoint_average_policy_value_aggregate.png
```

Random-search experiments additionally export:

```text
screening_configs.json
confirmation_configs.json
tables/screening_run_summaries.csv
tables/screening_curves.csv
tables/screening_config_summary.csv
tables/confirmation_run_summaries.csv
tables/confirmation_curves.csv
tables/confirmation_config_summary.csv
tables/confirmation_paired_differences_final_exploitability.csv
tables/confirmation_paired_differences_final_average_policy_value.csv
traces/<stage>/<config_label>/seed_<seed>_curves.csv
plots/screening_ranked_final_window_exploitability.png
plots/confirmation_final_exploitability.png
plots/confirmation_exploitability_by_iteration.png
plots/confirmation_final_average_policy_value.png
plots/confirmation_average_policy_value_by_iteration.png
```

Warm-start ablations additionally export:

```text
checkpoint_curves_by_arm.csv
seed_arm_summary.csv
aggregate_summary_by_arm.csv
aggregate_summary_by_arm.json
paired_differences_warm_minus_baseline.csv
paired_difference_summary.json
seed_<seed>/warm_start_resume/checkpoint/dream_checkpoint_iter_<iteration>.pt
plots/dream_warm_start_exploitability_by_iteration.png
plots/dream_warm_start_average_policy_value_by_iteration.png
plots/dream_warm_start_final_average_policy_value.png
plots/dream_warm_start_paired_final_exploitability_delta.png
```

Learning-rate schedule ablations additionally export:

```text
checkpoint_curves_by_variant.csv
seed_variant_summary.csv
aggregate_summary_by_variant.csv
aggregate_summary_by_variant.json
paired_differences_vs_baseline.csv
paired_difference_summary.json
multiseed_curves_by_variant.npz
seed_<seed>/<variant>/checkpoint_curves.csv
plots/dream_lr_schedule_exploitability_by_iteration.png
plots/dream_lr_schedule_final_exploitability.png
plots/dream_lr_schedule_average_policy_value_by_iteration.png
plots/dream_lr_schedule_final_average_policy_value.png
plots/dream_lr_schedule_paired_final_exploitability_delta.png
plots/dream_lr_schedule_learning_rate_schedules.png
```

Network-training budget ablations additionally export:

```text
checkpoint_curves_by_variant.csv
seed_variant_summary.csv
aggregate_summary_by_variant.csv
aggregate_summary_by_variant.json
paired_differences_vs_baseline.csv
paired_difference_summary.json
multiseed_curves_by_variant.npz
seed_<seed>/<variant>/checkpoint_curves.csv
plots/dream_baseline_budget_exploitability_by_iteration.png
plots/dream_baseline_budget_final_exploitability.png
plots/dream_baseline_budget_average_policy_value_by_iteration.png
plots/dream_baseline_budget_final_average_policy_value.png
plots/dream_baseline_budget_baseline_loss.png
plots/dream_baseline_budget_advantage_target_variance.png
plots/dream_baseline_budget_paired_final_exploitability_delta.png
```

Scalar-parameter ablations additionally export:

```text
checkpoint_curves_by_variant.csv
seed_variant_summary.csv
aggregate_summary_by_variant.csv
aggregate_summary_by_variant.json
paired_differences_vs_baseline.csv
paired_difference_summary.json
multiseed_curves_by_variant.npz
seed_<seed>/<variant>/checkpoint_curves.csv
plots/dream_epsilon_exploration_exploitability_by_iteration.png
plots/dream_epsilon_exploration_final_exploitability.png
plots/dream_epsilon_exploration_average_policy_value_by_iteration.png
plots/dream_epsilon_exploration_final_average_policy_value.png
plots/dream_epsilon_exploration_advantage_target_variance.png
plots/dream_epsilon_exploration_policy_entropy_mean.png
plots/dream_epsilon_exploration_paired_final_exploitability_delta.png
```

Trajectory-count ablations additionally export:

```text
checkpoint_curves_by_variant.csv
seed_variant_summary.csv
aggregate_summary_by_variant.csv
aggregate_summary_by_variant.json
paired_differences_vs_baseline.csv
paired_difference_summary.json
multiseed_curves_by_variant.npz
seed_<seed>/<variant>/checkpoint_curves.csv
plots/dream_trajectories_per_iteration_exploitability_by_iteration.png
plots/dream_trajectories_per_iteration_exploitability_by_nodes.png
plots/dream_trajectories_per_iteration_exploitability_by_sampled_trajectories.png
plots/dream_trajectories_per_iteration_average_policy_value_by_iteration.png
plots/dream_trajectories_per_iteration_average_policy_value_by_nodes.png
plots/dream_trajectories_per_iteration_average_policy_value_by_sampled_trajectories.png
plots/dream_trajectories_per_iteration_paired_sample_trajectory_auc_delta.png
```

Target-processing ablations additionally export:

```text
checkpoint_curves_by_variant.csv
seed_variant_summary.csv
aggregate_summary_by_variant.csv
aggregate_summary_by_variant.json
paired_differences_vs_baseline.csv
paired_difference_summary.json
multiseed_curves_by_variant.npz
seed_<seed>/<variant>/checkpoint_curves.csv
plots/dream_target_processing_exploitability_by_iteration.png
plots/dream_target_processing_processed_advantage_target_variance.png
plots/dream_target_processing_target_clip_fraction.png
plots/dream_target_processing_paired_final_exploitability_delta.png
```

Residual-network ablations additionally export the same architecture-ablation
files as the width/depth/capacity experiments, with network-treatment columns:

```text
checkpoint_curves_by_variant.csv
seed_variant_summary.csv
aggregate_summary_by_variant.csv
aggregate_summary_by_variant.json
paired_differences_vs_baseline.csv
paired_difference_summary.json
multiseed_curves_by_variant.npz
seed_<seed>/<variant>/checkpoint_curves.csv
plots/dream_residual_network_exploitability_by_iteration.png
plots/dream_residual_network_final_exploitability.png
plots/dream_residual_network_final_exploitability_by_parameters.png
plots/dream_residual_network_paired_final_exploitability_delta.png
```

Average-strategy weighting ablations additionally export the same matched-variant
files as the scalar ablations, with average-policy weighting columns:

```text
checkpoint_curves_by_variant.csv
seed_variant_summary.csv
aggregate_summary_by_variant.csv
aggregate_summary_by_variant.json
paired_differences_vs_baseline.csv
paired_difference_summary.json
multiseed_curves_by_variant.npz
seed_<seed>/<variant>/checkpoint_curves.csv
plots/dream_average_strategy_weighting_exploitability_by_iteration.png
plots/dream_average_strategy_weighting_average_policy_value_by_iteration.png
plots/dream_average_strategy_weighting_policy_loss.png
plots/dream_average_strategy_weighting_paired_final_exploitability_delta.png
```

Factorised advantage-head ablations additionally export the same
architecture-ablation files as the width/depth/capacity experiments, with
advantage-head treatment columns:

```text
checkpoint_curves_by_variant.csv
seed_variant_summary.csv
aggregate_summary_by_variant.csv
aggregate_summary_by_variant.json
paired_differences_vs_baseline.csv
paired_difference_summary.json
multiseed_curves_by_variant.npz
seed_<seed>/<variant>/checkpoint_curves.csv
plots/dream_factorised_advantage_head_exploitability_by_iteration.png
plots/dream_factorised_advantage_head_final_exploitability.png
plots/dream_factorised_advantage_head_final_exploitability_by_parameters.png
plots/dream_factorised_advantage_head_paired_final_exploitability_delta.png
```

## Notes for adding future experiments

When adding a new DREAM experiment, follow the same pattern as the baseline:

1. create a new folder under `experiments/leduc_poker/`;
2. include a `config.py`, `run.py`, and `README.md`;
3. hold the baseline protocol fixed except for the intended treatment variable;
4. use matched seeds where possible;
5. export the same core metrics and plots so the thesis results have a consistent look and feel.
