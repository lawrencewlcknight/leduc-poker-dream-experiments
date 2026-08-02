
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
│       ├── dream_factorised_advantage_head_ablation/   # Experiment 16
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_plain_network_depth_ablation/         # Experiment 17
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_layer_norm_network_ablation/          # Experiment 18
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_residual_layer_norm_network_ablation/ # Experiment 19
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_role_specific_capacity_ablation/      # Experiment 20
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_parallel_equivalence_ablation/        # Experiment 21
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_architecture_candidate_comparison/    # Experiment 22
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_candidate_epsilon_exploration_ablation/ # Experiment 23
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_candidate_traversal_budget_ablation/    # Experiment 24
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_candidate_strategy_replay_capacity_ablation/ # Experiment 25
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_candidate_baseline_replay_capacity_ablation/ # Experiment 26
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_candidate_policy_extraction_budget_ablation/ # Experiment 27
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_candidate_policy_extraction_cadence_ablation/ # Experiment 28
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_candidate_advantage_fitting_steps_ablation/ # Experiment 29
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_candidate_advantage_batch_size_ablation/ # Experiment 30
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_candidate_constant_learning_rate_ablation/ # Experiment 31
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_candidate_baseline_training_cadence_ablation/ # Experiment 32
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_candidate_long_node_epsilon_comparison/ # Experiment 33
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_candidate_epsilon_baseline50_comparison/ # Experiment 34
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_candidate_900_traversal_long_node/ # Experiment 35
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_candidate_epsilon020_900_traversal_long_node/ # Experiment 36
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_vectorized_baseline_equivalence/ # Experiment 37
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_candidate_epsilon_baseline50_long_node_comparison/ # Experiment 38
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_candidate_900_traversal_baseline1000_long_node/ # Experiment 39
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_candidate_epsilon020_900_traversal_baseline1000_long_node/ # Experiment 40
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_candidate_long_node_run/ # Experiment 41
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── dream_candidate_epsilon020_long_node_run/ # Experiment 42
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       └── dream_final_candidate_checkpoint_head_to_head/ # Experiment 43
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

### 17. DREAM plain-network depth reference ablation

[`experiments/leduc_poker/dream_plain_network_depth_ablation/`](experiments/leduc_poker/dream_plain_network_depth_ablation/README.md)

Compares plain MLP DREAM networks at fixed width `32` and hidden depths `2`,
`4`, and `8`. This is the depth-only control condition for the subsequent
LayerNorm experiments.

**Question:** how much of any apparent normalisation benefit is explained by
plain network depth rather than by LayerNorm itself?

### 18. DREAM LayerNorm network ablation

[`experiments/leduc_poker/dream_layer_norm_network_ablation/`](experiments/leduc_poker/dream_layer_norm_network_ablation/README.md)

Compares the baseline `plain_layers2_width32` variant with LayerNorm MLPs at
fixed width `32` and hidden depths `2`, `4`, and `8`. The same treatment is
applied to the average-policy, advantage, and learned-baseline networks within
each variant.

**Question:** does hidden-activation normalisation improve DREAM optimisation stability or final average-policy quality when all non-architecture settings are held fixed?

### 19. DREAM residual-LayerNorm network ablation

[`experiments/leduc_poker/dream_residual_layer_norm_network_ablation/`](experiments/leduc_poker/dream_residual_layer_norm_network_ablation/README.md)

Compares the baseline `plain_layers2_width32` variant with residual-LayerNorm
MLPs at fixed width `32` and hidden depths `2`, `4`, and `8`.

**Question:** does combining residual hidden blocks with LayerNorm improve
DREAM optimisation stability or final average-policy quality?

### 20. DREAM role-specific capacity ablation

[`experiments/leduc_poker/dream_role_specific_capacity_ablation/`](experiments/leduc_poker/dream_role_specific_capacity_ablation/README.md)

Tests whether the modest gains from the `3x64` architecture are driven mainly by the player-specific advantage networks. The policy and baseline networks remain fixed at `2x32` in the advantage-only arms, which are compared with matched `2x32` and all-network `3x64` references.

**Question:** can DREAM improve more efficiently by assigning extra capacity to advantage estimation rather than scaling every network family together?

### 21. DREAM sequential/parallel equivalence ablation

[`experiments/leduc_poker/dream_parallel_equivalence_ablation/`](experiments/leduc_poker/dream_parallel_equivalence_ablation/README.md)

Compares the current best documented DREAM configuration, uniform `3x64` policy, advantage, and baseline networks, under the current sequential traversal collector and a Ray-parallel collector with three workers. The learner remains central and the traversal/replay budgets are partitioned across workers rather than multiplied.

**Question:** does Ray-parallel DREAM preserve final policy quality while reducing traversal and end-to-end runtime?

### 22. DREAM architecture-candidate comparison

[`experiments/leduc_poker/dream_architecture_candidate_comparison/`](experiments/leduc_poker/dream_architecture_candidate_comparison/README.md)

Compares the original all-network `2x32` DREAM baseline with the architecture-selected candidate from the role-specific capacity analysis: `2x32` average-policy and learned-baseline networks with `2x128` player-specific advantage networks. The default run uses the five baseline seeds and the original DREAM training protocol.

**Question:** does the architecture-selected candidate improve final exploitability, final-window exploitability, and node-normalised sample efficiency relative to the original DREAM baseline under a matched five-seed protocol?

### 23. DREAM candidate-architecture epsilon-exploration ablation

[`experiments/leduc_poker/dream_candidate_epsilon_exploration_ablation/`](experiments/leduc_poker/dream_candidate_epsilon_exploration_ablation/README.md)

Retunes the epsilon-mixed traversal policy after adopting the architecture-selected DREAM candidate as the new baseline. The policy and baseline networks remain `2x32`, the player-specific advantage networks use `2x128`, and the epsilon grid is `0.06`, `0.10`, `0.15`, and `0.20`.

**Question:** does the previously favourable `epsilon=0.10` remain best under the new architecture, or does a higher exploration rate further improve exploitability and node-normalised sample efficiency?

### 24. DREAM candidate traversal-budget ablation

[`experiments/leduc_poker/dream_candidate_traversal_budget_ablation/`](experiments/leduc_poker/dream_candidate_traversal_budget_ablation/README.md)

Starts from the architecture-selected candidate and varies only `num_traversals`, comparing `160`, `320`, and `480` traversals per player per DREAM iteration.

**Question:** was the strong random-search signal mainly caused by reducing outcome-sampling target noise, and does any gain remain when exploitability is measured by nodes touched?

### 25. DREAM candidate strategy replay-capacity ablation

[`experiments/leduc_poker/dream_candidate_strategy_replay_capacity_ablation/`](experiments/leduc_poker/dream_candidate_strategy_replay_capacity_ablation/README.md)

Starts from the architecture-selected candidate and varies only `strategy_memory_capacity`, comparing `1e6`, `5e5`, and `1e5` average-policy replay entries.

**Question:** does the neural average-policy extractor benefit from fresher strategy replay rather than a larger reservoir containing more early-training behaviour?

### 26. DREAM candidate learned-baseline replay-capacity ablation

[`experiments/leduc_poker/dream_candidate_baseline_replay_capacity_ablation/`](experiments/leduc_poker/dream_candidate_baseline_replay_capacity_ablation/README.md)

Starts from the architecture-selected candidate and varies only `baseline_memory_capacity`, comparing `1e6`, `5e5`, and `1e5` learned-baseline replay entries.

**Question:** does the DREAM control variate need fresher replay to track the current sampling distribution and reduce target variance?

### 27. DREAM candidate policy-extraction budget ablation

[`experiments/leduc_poker/dream_candidate_policy_extraction_budget_ablation/`](experiments/leduc_poker/dream_candidate_policy_extraction_budget_ablation/README.md)

Starts from the architecture-selected candidate and varies only `policy_network_train_steps`, comparing `50`, `100`, `200`, and `400` average-policy update steps per training event.

**Question:** is exploitability limited by underfitting or overfitting the evaluated neural average-policy network?

### 28. DREAM candidate policy-extraction cadence ablation

[`experiments/leduc_poker/dream_candidate_policy_extraction_cadence_ablation/`](experiments/leduc_poker/dream_candidate_policy_extraction_cadence_ablation/README.md)

Starts from the architecture-selected candidate and varies only `policy_network_train_every`, comparing policy extraction every `10`, `25`, and `50` DREAM iterations.

**Question:** does the average-policy network benefit from reduced lag behind the evolving regret-induced strategy distribution?

### 29. DREAM candidate advantage-fitting steps ablation

[`experiments/leduc_poker/dream_candidate_advantage_fitting_steps_ablation/`](experiments/leduc_poker/dream_candidate_advantage_fitting_steps_ablation/README.md)

Starts from the architecture-selected candidate and varies only `advantage_network_train_steps`, comparing `25`, `50`, `100`, and `200` updates per player per DREAM iteration.

**Question:** are the larger `2x128` advantage networks under-optimised, or do fewer updates better regularise noisy sampled regret targets?

### 30. DREAM candidate advantage batch-size ablation

[`experiments/leduc_poker/dream_candidate_advantage_batch_size_ablation/`](experiments/leduc_poker/dream_candidate_advantage_batch_size_ablation/README.md)

Starts from the architecture-selected candidate and varies only `batch_size_advantage`, comparing minibatches of `512`, `1024`, and `2048`.

**Question:** does DREAM advantage fitting benefit more from stochastic regularisation or lower minibatch-gradient variance?

### 31. DREAM candidate constant learning-rate ablation

[`experiments/leduc_poker/dream_candidate_constant_learning_rate_ablation/`](experiments/leduc_poker/dream_candidate_constant_learning_rate_ablation/README.md)

Starts from the architecture-selected candidate and varies only the constant optimiser learning rate, comparing `0.001`, `0.003`, and `0.006`.

**Question:** does the candidate baseline prefer a slower or faster fixed optimiser step size once learning-rate decay has already been ruled out?

### 32. DREAM candidate baseline-training cadence ablation

[`experiments/leduc_poker/dream_candidate_baseline_training_cadence_ablation/`](experiments/leduc_poker/dream_candidate_baseline_training_cadence_ablation/README.md)

Starts from the architecture-selected candidate and varies only `baseline_network_train_every`, comparing learned-baseline training every `1`, `5`, `10`, `25`, and `50` DREAM iterations. The `every 1` comparator is reused from the tracked Experiment 22 candidate artifact by default.

**Question:** does DREAM need to refit the learned Q-baseline on every iteration, or can a stale control variate retain most of the variance-reduction benefit while sharply reducing wall-clock time?

### 33. DREAM long-node candidate epsilon comparison

[`experiments/leduc_poker/dream_candidate_long_node_epsilon_comparison/`](experiments/leduc_poker/dream_candidate_long_node_epsilon_comparison/README.md)

Trains the architecture-selected candidate and the same configuration with `epsilon=0.20` over approximately `15m` nodes touched per seed.

**Question:** does the stronger exploration setting remain preferable under a substantially longer training horizon?

### 34. DREAM candidate epsilon comparison with sparse baseline training

[`experiments/leduc_poker/dream_candidate_epsilon_baseline50_comparison/`](experiments/leduc_poker/dream_candidate_epsilon_baseline50_comparison/README.md)

Starts from the architecture-selected candidate, sets `baseline_network_train_every=50` in both arms, and compares `epsilon=0.06` against `epsilon=0.20` over three seeds.

**Question:** does the higher-exploration candidate remain preferable when the learned Q-baseline is refit only sparsely?

### 35. DREAM candidate 900-traversal long-node cadence comparison

[`experiments/leduc_poker/dream_candidate_900_traversal_long_node/`](experiments/leduc_poker/dream_candidate_900_traversal_long_node/README.md)

Trains two vectorized DREAM variants with the architecture-selected candidate settings and `900` outcome-sampling traversals per player per iteration for `1,300` DREAM iterations, targeting roughly `15m` nodes touched per seed. The comparator preserves learned-baseline training every iteration; the treatment changes only the learned-baseline cadence to every `50` iterations.

**Question:** does the higher per-iteration traversal budget from the DREAM paper remain effective when learned-baseline refitting is made sparse enough to reduce the dominant training cost?

### 36. DREAM epsilon-0.20 900-traversal long-node cadence comparison

[`experiments/leduc_poker/dream_candidate_epsilon020_900_traversal_long_node/`](experiments/leduc_poker/dream_candidate_epsilon020_900_traversal_long_node/README.md)

Trains two vectorized DREAM variants with `epsilon=0.20`, the architecture-selected candidate settings, and `900` outcome-sampling traversals per player per iteration for `1,300` DREAM iterations, again targeting roughly `15m` nodes touched per seed. The comparator preserves learned-baseline training every iteration; the treatment changes only the learned-baseline cadence to every `50` iterations.

**Question:** does the stronger exploration setting remain beneficial under the higher per-iteration traversal budget when learned-baseline refitting is made sparse?

### 37. DREAM vectorized-baseline equivalence check

[`experiments/leduc_poker/dream_vectorized_baseline_equivalence/`](experiments/leduc_poker/dream_vectorized_baseline_equivalence/README.md)

Reruns the Experiment 22 architecture-selected candidate with tensorized learned-baseline replay and the vectorized baseline learner, while reusing the archived Experiment 22 candidate outputs as the original-implementation comparator.

**Question:** does the vectorized learned-baseline implementation preserve exploitability and average-policy value while reducing wall-clock and learned-baseline training time?

### 38. DREAM candidate epsilon comparison with sparse baseline training at long-node horizon

[`experiments/leduc_poker/dream_candidate_epsilon_baseline50_long_node_comparison/`](experiments/leduc_poker/dream_candidate_epsilon_baseline50_long_node_comparison/README.md)

Extends Experiment 34 to the long-node horizon using the vectorized learned-baseline implementation. Both arms train the learned baseline every `50` DREAM iterations and keep `160` traversals per player per iteration; the comparison is between `epsilon=0.06` and `epsilon=0.20` over `7,500` iterations, targeting roughly `15m` nodes per seed.

**Question:** does the higher-exploration sparse-baseline configuration remain preferable when trained to the long-node budget?

### 39. DREAM candidate 900-traversal baseline-1000 long-node run

[`experiments/leduc_poker/dream_candidate_900_traversal_baseline1000_long_node/`](experiments/leduc_poker/dream_candidate_900_traversal_baseline1000_long_node/README.md)

Starts from the Experiment 22 architecture-selected candidate and uses the vectorized implementation with `900` traversals per player per iteration and `1000` learned-baseline minibatches per player per iteration. The run keeps learned-baseline training on every DREAM iteration, uses three seeds, and trains for `1,300` iterations, targeting roughly `15m` nodes per seed.

**Question:** does the paper-style high traversal and high baseline-fitting budget produce a stronger long-node DREAM policy under the vectorized implementation?

### 40. DREAM epsilon-0.20 900-traversal baseline-1000 long-node run

[`experiments/leduc_poker/dream_candidate_epsilon020_900_traversal_baseline1000_long_node/`](experiments/leduc_poker/dream_candidate_epsilon020_900_traversal_baseline1000_long_node/README.md)

Keeps the Experiment 39 paper-style traversal and baseline-fitting budget but changes only the exploration rate to `epsilon=0.20`. The run uses three seeds and trains for `1,300` iterations, targeting roughly `15m` nodes per seed.

**Question:** does the higher-exploration setting improve the paper-style high traversal and high learned-baseline fitting budget?

### 41. DREAM candidate long-node run

[`experiments/leduc_poker/dream_candidate_long_node_run/`](experiments/leduc_poker/dream_candidate_long_node_run/README.md)

Trains the Experiment 22 architecture-selected candidate as a dedicated single-arm vectorized run over `7,300` iterations with `160` traversals per player per iteration, targeting roughly `15m` nodes per seed. Experiment 33 already includes this configuration as its `epsilon=0.06` arm, but that experiment also trains the `epsilon=0.20` treatment.

**Question:** how does the Experiment 22 candidate behave when trained alone to the long-node budget without spending compute on an additional comparison arm?

### 42. DREAM epsilon-0.20 candidate long-node run

[`experiments/leduc_poker/dream_candidate_epsilon020_long_node_run/`](experiments/leduc_poker/dream_candidate_epsilon020_long_node_run/README.md)

Replicates Experiment 41 and changes only the exploration rate to `epsilon=0.20`. The run uses the vectorized implementation, `160` traversals per player per iteration, five seeds, and `7,300` iterations, targeting roughly `15m` nodes per seed.

**Question:** does the high-exploration version of the Experiment 22 candidate improve when trained alone to the long-node budget?

### 43. DREAM final-candidate checkpoint head-to-head

[`experiments/leduc_poker/dream_final_candidate_checkpoint_head_to_head/`](experiments/leduc_poker/dream_final_candidate_checkpoint_head_to_head/README.md)

Trains the best DREAM configuration selected from Experiment 38 once per seed and saves average-policy snapshots at `1,500`, `3,000`, `4,500`, `6,000`, and `7,500` iterations, corresponding to approximately `3m`, `6m`, `9m`, `12m`, and `15m` nodes. Every checkpoint is then evaluated against every other checkpoint by exact seat-averaged OpenSpiel expected value, with seed-level inference matching the Deep CFR temporal head-to-head experiment.

**Question:** does the long-horizon exploitability improvement in the selected DREAM configuration correspond to progressively stronger direct-play performance?

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

# Experiment 17 — plain-network depth reference ablation
python -m experiments.leduc_poker.dream_plain_network_depth_ablation.run

# Experiment 18 — LayerNorm network ablation
python -m experiments.leduc_poker.dream_layer_norm_network_ablation.run

# Experiment 19 — residual-LayerNorm network ablation
python -m experiments.leduc_poker.dream_residual_layer_norm_network_ablation.run

# Experiment 20 — role-specific capacity ablation
python -m experiments.leduc_poker.dream_role_specific_capacity_ablation.run

# Experiment 21 — sequential versus Ray-parallel equivalence ablation
python -m experiments.leduc_poker.dream_parallel_equivalence_ablation.run

# Experiment 22 — architecture-candidate comparison
python -m experiments.leduc_poker.dream_architecture_candidate_comparison.run

# Experiment 23 — candidate-architecture epsilon-exploration ablation
python -m experiments.leduc_poker.dream_candidate_epsilon_exploration_ablation.run

# Experiment 24 — candidate traversal-budget ablation
python -m experiments.leduc_poker.dream_candidate_traversal_budget_ablation.run

# Experiment 25 — candidate strategy replay-capacity ablation
python -m experiments.leduc_poker.dream_candidate_strategy_replay_capacity_ablation.run

# Experiment 26 — candidate learned-baseline replay-capacity ablation
python -m experiments.leduc_poker.dream_candidate_baseline_replay_capacity_ablation.run

# Experiment 27 — candidate policy-extraction budget ablation
python -m experiments.leduc_poker.dream_candidate_policy_extraction_budget_ablation.run

# Experiment 28 — candidate policy-extraction cadence ablation
python -m experiments.leduc_poker.dream_candidate_policy_extraction_cadence_ablation.run

# Experiment 29 — candidate advantage-fitting steps ablation
python -m experiments.leduc_poker.dream_candidate_advantage_fitting_steps_ablation.run

# Experiment 30 — candidate advantage batch-size ablation
python -m experiments.leduc_poker.dream_candidate_advantage_batch_size_ablation.run

# Experiment 31 — candidate constant learning-rate ablation
python -m experiments.leduc_poker.dream_candidate_constant_learning_rate_ablation.run

# Experiment 32 — candidate baseline-training cadence ablation
python -m experiments.leduc_poker.dream_candidate_baseline_training_cadence_ablation.run

# Experiment 33 — long-node candidate epsilon comparison
python -m experiments.leduc_poker.dream_candidate_long_node_epsilon_comparison.run

# Experiment 34 — candidate epsilon comparison with baseline cadence 50
python -m experiments.leduc_poker.dream_candidate_epsilon_baseline50_comparison.run

# Experiment 35 — 900-traversal long-node baseline-cadence comparison
python -m experiments.leduc_poker.dream_candidate_900_traversal_long_node.run

# Experiment 36 — epsilon-0.20 900-traversal long-node baseline-cadence comparison
python -m experiments.leduc_poker.dream_candidate_epsilon020_900_traversal_long_node.run

# Experiment 37 — vectorized-baseline equivalence check
python -m experiments.leduc_poker.dream_vectorized_baseline_equivalence.run

# Experiment 38 — long-node epsilon comparison with baseline cadence 50
python -m experiments.leduc_poker.dream_candidate_epsilon_baseline50_long_node_comparison.run

# Experiment 39 — 900-traversal baseline-1000 long-node run
python -m experiments.leduc_poker.dream_candidate_900_traversal_baseline1000_long_node.run

# Experiment 40 — epsilon-0.20 900-traversal baseline-1000 long-node run
python -m experiments.leduc_poker.dream_candidate_epsilon020_900_traversal_baseline1000_long_node.run

# Experiment 41 — Experiment 22 candidate long-node run
python -m experiments.leduc_poker.dream_candidate_long_node_run.run

# Experiment 42 — epsilon-0.20 candidate long-node run
python -m experiments.leduc_poker.dream_candidate_epsilon020_long_node_run.run

# Experiment 43 — final-candidate temporal checkpoint head-to-head
python -m experiments.leduc_poker.dream_final_candidate_checkpoint_head_to_head.run
```

Experiments 23--32 and 37 reuse the tracked Experiment 22
candidate-architecture baseline artifact as their comparator by default. This
avoids retraining the unchanged baseline arm in each ablation. Pass
`--train-baseline` only when you intentionally want to regenerate the comparator
for a debugging run. The full commands for these ablations do not pass
`--train-baseline`, so they reuse the cached comparator.

Experiments 33--36 and 38--42 are different: they intentionally train all configured
variants and do not reuse a fixed baseline artifact.

Some smoke-test examples below pass `--train-baseline` explicitly. This is only
to exercise both training arms under a tiny three-iteration budget; it is not the
default behaviour for full experiments.

To run quick smoke tests for later DREAM ablations on GCP, use the Batch
submission script. These commands do not require the repository Python
dependencies to be installed locally; they only require the Google Cloud CLI,
the GCP environment variables from
[`docs/GCP_BATCH_EXPERIMENTS.md`](docs/GCP_BATCH_EXPERIMENTS.md), and a local
`python3` capable of running the submission helper.

```bash
# Submit Experiments 13-17 together with one shared timestamp.
./gcp/submit_recent_ablation_smoke_tests.sh

# Submit only Experiments 16-17 together with one shared timestamp.
./gcp/submit_latest_ablation_smoke_tests.sh

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

# Leduc Experiment 17 — plain-network depth reference smoke test on GCP
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp17-plain-depth-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_plain_network_depth_ablation.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --variants plain_layers2_width32,plain_layers4_width32 \
    --output-root outputs/cloud/smoke/leduc_dream_plain_network_depth_ablation" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"

# Leduc Experiment 18 — LayerNorm network ablation smoke test on GCP
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp18-layer-norm-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_layer_norm_network_ablation.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --variants plain_layers2_width32,layer_norm_layers2_width32 \
    --output-root outputs/cloud/smoke/leduc_dream_layer_norm_network_ablation" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"

# Leduc Experiment 19 — residual-LayerNorm network ablation smoke test on GCP
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp19-residual-layer-norm-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_residual_layer_norm_network_ablation.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --variants plain_layers2_width32,residual_layer_norm_layers2_width32 \
    --output-root outputs/cloud/smoke/leduc_dream_residual_layer_norm_network_ablation" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"

# Leduc Experiment 20 — role-specific capacity ablation smoke test on GCP
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp20-role-capacity-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_role_specific_capacity_ablation.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --variants all_2x32_reference,advantage_3x64_policy_baseline_2x32,advantage_2x128_policy_baseline_2x32,all_3x64_reference \
    --output-root outputs/cloud/smoke/leduc_dream_role_specific_capacity_ablation" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"

# Leduc Experiment 21 — sequential/parallel equivalence smoke test on GCP
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp21-parallel-equivalence-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_parallel_equivalence_ablation.run \
    --seeds 1234 \
    --variant-ids dream_3x64_sequential,dream_3x64_ray_parallel \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --batch-size-advantage 1 \
    --batch-size-strategy 1 \
    --batch-size-baseline 1 \
    --output-root outputs/cloud/smoke/leduc_dream_parallel_equivalence_ablation" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"

# Leduc Experiment 22 — architecture-candidate comparison smoke test on GCP
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp22-architecture-candidate-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_architecture_candidate_comparison.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --variants baseline_all_2x32,candidate_advantage_2x128_policy_baseline_2x32 \
    --output-root outputs/cloud/smoke/leduc_dream_architecture_candidate_comparison" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"

# Leduc Experiment 23 — candidate-architecture epsilon smoke test on GCP
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

# Leduc Experiment 24 — candidate traversal-budget smoke test on GCP
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp24-traversal-budget-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_traversal_budget_ablation.run \
    --seeds 1234 \
    --iterations 3 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --variants candidate_traversals_160_baseline,candidate_traversals_320 \
    --train-baseline \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_traversal_budget_ablation" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"

# Leduc Experiment 25 — candidate strategy replay-capacity smoke test on GCP
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp25-strategy-replay-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_strategy_replay_capacity_ablation.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --variants candidate_strategy_memory_1m_baseline,candidate_strategy_memory_100k \
    --train-baseline \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_strategy_replay_capacity_ablation" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"

# Leduc Experiment 26 — candidate learned-baseline replay-capacity smoke test on GCP
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp26-baseline-replay-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_baseline_replay_capacity_ablation.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --variants candidate_baseline_memory_1m_baseline,candidate_baseline_memory_100k \
    --train-baseline \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_baseline_replay_capacity_ablation" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"

# Leduc Experiment 27 — candidate policy-extraction budget smoke test on GCP
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp27-policy-budget-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_policy_extraction_budget_ablation.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --variants candidate_policy_steps_100_baseline,candidate_policy_steps_200 \
    --train-baseline \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_policy_extraction_budget_ablation" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"

# Leduc Experiment 28 — candidate policy-extraction cadence smoke test on GCP
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp28-policy-cadence-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_policy_extraction_cadence_ablation.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --variants candidate_policy_every_25_baseline,candidate_policy_every_10 \
    --train-baseline \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_policy_extraction_cadence_ablation" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"

# Leduc Experiment 29 — candidate advantage-fitting steps smoke test on GCP
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp29-advantage-steps-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_advantage_fitting_steps_ablation.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --variants candidate_advantage_steps_50_baseline,candidate_advantage_steps_25 \
    --train-baseline \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_advantage_fitting_steps_ablation" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"

# Leduc Experiment 30 — candidate advantage batch-size smoke test on GCP
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp30-advantage-batch-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_advantage_batch_size_ablation.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --variants candidate_advantage_batch_1024_baseline,candidate_advantage_batch_512 \
    --train-baseline \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_advantage_batch_size_ablation" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"

# Leduc Experiment 31 — candidate constant learning-rate smoke test on GCP
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp31-constant-lr-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_constant_learning_rate_ablation.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --variants candidate_learning_rate_0_003_baseline,candidate_learning_rate_0_006 \
    --train-baseline \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_constant_learning_rate_ablation" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"

# Leduc Experiment 32 — candidate baseline-training cadence smoke test on GCP
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp32-baseline-cadence-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_baseline_training_cadence_ablation.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --variants candidate_baseline_every_1_baseline,candidate_baseline_every_5 \
    --train-baseline \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_baseline_training_cadence_ablation" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"

# Leduc Experiment 33 — long-node candidate epsilon smoke test on GCP
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp33-long-node-epsilon-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_long_node_epsilon_comparison.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_long_node_epsilon_comparison" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"

# Leduc Experiment 34 — candidate epsilon baseline-cadence-50 smoke test on GCP
./gcp/submit_batch_experiment.sh \
  "ld-exp34-eb50-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_epsilon_baseline50_comparison.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_epsilon_baseline50_comparison" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"

# Leduc Experiment 35 — 900-traversal long-node baseline-cadence smoke test on GCP
./gcp/submit_batch_experiment.sh \
  "ld-exp35-t900-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_900_traversal_long_node.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_900_traversal_long_node" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"

# Leduc Experiment 36 — epsilon-0.20 900-traversal long-node baseline-cadence smoke test on GCP
./gcp/submit_batch_experiment.sh \
  "ld-exp36-e02t900-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_epsilon020_900_traversal_long_node.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_epsilon020_900_traversal_long_node" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"

# Leduc Experiment 37 — vectorized-baseline equivalence smoke test on GCP
./gcp/submit_batch_experiment.sh \
  "ld-exp37-vbleq-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_vectorized_baseline_equivalence.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --output-root outputs/cloud/smoke/leduc_dream_vectorized_baseline_equivalence" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"

# Leduc Experiment 38 — long-node epsilon baseline-cadence-50 smoke test on GCP
./gcp/submit_batch_experiment.sh \
  "ld-exp38-eb50ln-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_epsilon_baseline50_long_node_comparison.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_epsilon_baseline50_long_node_comparison" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"

# Leduc Experiment 39 — 900-traversal baseline-1000 long-node smoke test on GCP
./gcp/submit_batch_experiment.sh \
  "ld-exp39-t900b1000-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_900_traversal_baseline1000_long_node.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_900_traversal_baseline1000_long_node" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"

# Leduc Experiment 40 — epsilon-0.20 900-traversal baseline-1000 long-node smoke test on GCP
./gcp/submit_batch_experiment.sh \
  "ld-exp40-e02t900b1000-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_epsilon020_900_traversal_baseline1000_long_node.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_epsilon020_900_traversal_baseline1000_long_node" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"

# Leduc Experiment 41 — Experiment 22 candidate long-node smoke test on GCP
./gcp/submit_batch_experiment.sh \
  "ld-exp41-longnode-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_long_node_run.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_long_node_run" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"

# Leduc Experiment 42 — epsilon-0.20 candidate long-node smoke test on GCP
./gcp/submit_batch_experiment.sh \
  "ld-exp42-e02-longnode-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_epsilon020_long_node_run.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_epsilon020_long_node_run" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"

# Leduc Experiment 43 — final-candidate checkpoint head-to-head smoke test on GCP
./gcp/submit_batch_experiment.sh \
  "ld-exp43-final-h2h-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_final_candidate_checkpoint_head_to_head.run \
    --seeds 1234 \
    --iterations 10 \
    --checkpoint-schedule 2,4,6,8,10 \
    --traversals 4 \
    --evaluation-interval 2 \
    --policy-network-train-every 2 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --baseline-network-train-every 2 \
    --policy-network-layers 8,8 \
    --advantage-network-layers 8,8 \
    --baseline-network-layers 8,8 \
    --batch-size-advantage 2 \
    --batch-size-strategy 2 \
    --batch-size-baseline 2 \
    --memory-capacity 256 \
    --output-root outputs/cloud/smoke/leduc_dream_final_candidate_checkpoint_head_to_head" \
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

python -m experiments.leduc_poker.dream_plain_network_depth_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants plain_layers2_width32,plain_layers4_width32 \
  --output-root outputs/smoke_tests/dream_plain_network_depth_ablation

python -m experiments.leduc_poker.dream_layer_norm_network_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants plain_layers2_width32,layer_norm_layers2_width32 \
  --output-root outputs/smoke_tests/dream_layer_norm_network_ablation

python -m experiments.leduc_poker.dream_residual_layer_norm_network_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants plain_layers2_width32,residual_layer_norm_layers2_width32 \
  --output-root outputs/smoke_tests/dream_residual_layer_norm_network_ablation

# Leduc Experiment 20 — role-specific capacity ablation smoke test
python -m experiments.leduc_poker.dream_role_specific_capacity_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants all_2x32_reference,advantage_3x64_policy_baseline_2x32,advantage_2x128_policy_baseline_2x32,all_3x64_reference \
  --output-root outputs/smoke_tests/dream_role_specific_capacity_ablation

# Leduc Experiment 21 — sequential/parallel equivalence smoke test
python -m experiments.leduc_poker.dream_parallel_equivalence_ablation.run \
  --seeds 1234 \
  --variant-ids dream_3x64_sequential,dream_3x64_ray_parallel \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --batch-size-advantage 1 \
  --batch-size-strategy 1 \
  --batch-size-baseline 1 \
  --output-root outputs/smoke_tests/dream_parallel_equivalence_ablation

# Leduc Experiment 22 — architecture-candidate comparison smoke test
python -m experiments.leduc_poker.dream_architecture_candidate_comparison.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants baseline_all_2x32,candidate_advantage_2x128_policy_baseline_2x32 \
  --output-root outputs/smoke_tests/dream_architecture_candidate_comparison

# Leduc Experiment 23 — candidate-architecture epsilon smoke test
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

# Leduc Experiment 24 — candidate traversal-budget smoke test
python -m experiments.leduc_poker.dream_candidate_traversal_budget_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants candidate_traversals_160_baseline,candidate_traversals_320 \
  --train-baseline \
  --output-root outputs/smoke_tests/dream_candidate_traversal_budget_ablation

# Leduc Experiment 25 — candidate strategy replay-capacity smoke test
python -m experiments.leduc_poker.dream_candidate_strategy_replay_capacity_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants candidate_strategy_memory_1m_baseline,candidate_strategy_memory_100k \
  --train-baseline \
  --output-root outputs/smoke_tests/dream_candidate_strategy_replay_capacity_ablation

# Leduc Experiment 26 — candidate learned-baseline replay-capacity smoke test
python -m experiments.leduc_poker.dream_candidate_baseline_replay_capacity_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants candidate_baseline_memory_1m_baseline,candidate_baseline_memory_100k \
  --train-baseline \
  --output-root outputs/smoke_tests/dream_candidate_baseline_replay_capacity_ablation

# Leduc Experiment 27 — candidate policy-extraction budget smoke test
python -m experiments.leduc_poker.dream_candidate_policy_extraction_budget_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants candidate_policy_steps_100_baseline,candidate_policy_steps_200 \
  --train-baseline \
  --output-root outputs/smoke_tests/dream_candidate_policy_extraction_budget_ablation

# Leduc Experiment 28 — candidate policy-extraction cadence smoke test
python -m experiments.leduc_poker.dream_candidate_policy_extraction_cadence_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --variants candidate_policy_every_25_baseline,candidate_policy_every_10 \
  --train-baseline \
  --output-root outputs/smoke_tests/dream_candidate_policy_extraction_cadence_ablation

# Leduc Experiment 29 — candidate advantage-fitting steps smoke test
python -m experiments.leduc_poker.dream_candidate_advantage_fitting_steps_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants candidate_advantage_steps_50_baseline,candidate_advantage_steps_25 \
  --train-baseline \
  --output-root outputs/smoke_tests/dream_candidate_advantage_fitting_steps_ablation

# Leduc Experiment 30 — candidate advantage batch-size smoke test
python -m experiments.leduc_poker.dream_candidate_advantage_batch_size_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants candidate_advantage_batch_1024_baseline,candidate_advantage_batch_512 \
  --train-baseline \
  --output-root outputs/smoke_tests/dream_candidate_advantage_batch_size_ablation

# Leduc Experiment 31 — candidate constant learning-rate smoke test
python -m experiments.leduc_poker.dream_candidate_constant_learning_rate_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants candidate_learning_rate_0_003_baseline,candidate_learning_rate_0_006 \
  --train-baseline \
  --output-root outputs/smoke_tests/dream_candidate_constant_learning_rate_ablation

# Leduc Experiment 32 — candidate baseline-training cadence smoke test
python -m experiments.leduc_poker.dream_candidate_baseline_training_cadence_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants candidate_baseline_every_1_baseline,candidate_baseline_every_5 \
  --train-baseline \
  --output-root outputs/smoke_tests/dream_candidate_baseline_training_cadence_ablation

# Leduc Experiment 33 — long-node candidate epsilon smoke test
python -m experiments.leduc_poker.dream_candidate_long_node_epsilon_comparison.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests/dream_candidate_long_node_epsilon_comparison

# Leduc Experiment 34 — candidate epsilon baseline-cadence-50 smoke test
python -m experiments.leduc_poker.dream_candidate_epsilon_baseline50_comparison.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests/dream_candidate_epsilon_baseline50_comparison

# Leduc Experiment 35 — 900-traversal long-node baseline-cadence smoke test
python -m experiments.leduc_poker.dream_candidate_900_traversal_long_node.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests/dream_candidate_900_traversal_long_node

# Leduc Experiment 36 — epsilon-0.20 900-traversal long-node baseline-cadence smoke test
python -m experiments.leduc_poker.dream_candidate_epsilon020_900_traversal_long_node.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests/dream_candidate_epsilon020_900_traversal_long_node

# Leduc Experiment 37 — vectorized-baseline equivalence smoke test
python -m experiments.leduc_poker.dream_vectorized_baseline_equivalence.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests/dream_vectorized_baseline_equivalence

# Leduc Experiment 38 — long-node epsilon baseline-cadence-50 smoke test
python -m experiments.leduc_poker.dream_candidate_epsilon_baseline50_long_node_comparison.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests/dream_candidate_epsilon_baseline50_long_node_comparison

# Leduc Experiment 39 — 900-traversal baseline-1000 long-node smoke test
python -m experiments.leduc_poker.dream_candidate_900_traversal_baseline1000_long_node.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests/dream_candidate_900_traversal_baseline1000_long_node

# Leduc Experiment 40 — epsilon-0.20 900-traversal baseline-1000 long-node smoke test
python -m experiments.leduc_poker.dream_candidate_epsilon020_900_traversal_baseline1000_long_node.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests/dream_candidate_epsilon020_900_traversal_baseline1000_long_node

# Leduc Experiment 41 — Experiment 22 candidate long-node smoke test
python -m experiments.leduc_poker.dream_candidate_long_node_run.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests/dream_candidate_long_node_run

# Leduc Experiment 42 — epsilon-0.20 candidate long-node smoke test
python -m experiments.leduc_poker.dream_candidate_epsilon020_long_node_run.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests/dream_candidate_epsilon020_long_node_run

# Leduc Experiment 43 — final-candidate checkpoint head-to-head smoke test
python -m experiments.leduc_poker.dream_final_candidate_checkpoint_head_to_head.run \
  --seeds 1234 \
  --iterations 10 \
  --checkpoint-schedule 2,4,6,8,10 \
  --traversals 4 \
  --evaluation-interval 2 \
  --policy-network-train-every 2 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --baseline-network-train-every 2 \
  --policy-network-layers 8,8 \
  --advantage-network-layers 8,8 \
  --baseline-network-layers 8,8 \
  --batch-size-advantage 2 \
  --batch-size-strategy 2 \
  --batch-size-baseline 2 \
  --memory-capacity 256 \
  --output-root outputs/smoke_tests/dream_final_candidate_checkpoint_head_to_head
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

python -m experiments.leduc_poker.dream_plain_network_depth_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants plain_layers2_width32,plain_layers4_width32 \
  --output-root outputs/smoke_tests/dream_plain_network_depth_ablation

python -m experiments.leduc_poker.dream_layer_norm_network_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants plain_layers2_width32,layer_norm_layers2_width32 \
  --output-root outputs/smoke_tests/dream_layer_norm_network_ablation

python -m experiments.leduc_poker.dream_residual_layer_norm_network_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants plain_layers2_width32,residual_layer_norm_layers2_width32 \
  --output-root outputs/smoke_tests/dream_residual_layer_norm_network_ablation

# Leduc Experiment 20 — role-specific capacity ablation smoke test
python -m experiments.leduc_poker.dream_role_specific_capacity_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants all_2x32_reference,advantage_3x64_policy_baseline_2x32,advantage_2x128_policy_baseline_2x32,all_3x64_reference \
  --output-root outputs/smoke_tests/dream_role_specific_capacity_ablation

# Leduc Experiment 21 — sequential/parallel equivalence smoke test
python -m experiments.leduc_poker.dream_parallel_equivalence_ablation.run \
  --seeds 1234 \
  --variant-ids dream_3x64_sequential,dream_3x64_ray_parallel \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --batch-size-advantage 1 \
  --batch-size-strategy 1 \
  --batch-size-baseline 1 \
  --output-root outputs/smoke_tests/dream_parallel_equivalence_ablation

# Leduc Experiment 22 — architecture-candidate comparison smoke test
python -m experiments.leduc_poker.dream_architecture_candidate_comparison.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants baseline_all_2x32,candidate_advantage_2x128_policy_baseline_2x32 \
  --output-root outputs/smoke_tests/dream_architecture_candidate_comparison

# Leduc Experiment 23 — candidate-architecture epsilon smoke test
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

Layer-normalisation network ablations additionally export the same
architecture-ablation files as the width/depth/capacity experiments, with
network-treatment columns:

```text
checkpoint_curves_by_variant.csv
seed_variant_summary.csv
aggregate_summary_by_variant.csv
aggregate_summary_by_variant.json
paired_differences_vs_baseline.csv
paired_difference_summary.json
multiseed_curves_by_variant.npz
seed_<seed>/<variant>/checkpoint_curves.csv
plots/dream_layer_norm_network_exploitability_by_iteration.png
plots/dream_layer_norm_network_final_exploitability.png
plots/dream_layer_norm_network_final_exploitability_by_parameters.png
plots/dream_layer_norm_network_paired_final_exploitability_delta.png
```

## Notes for adding future experiments

When adding a new DREAM experiment, follow the same pattern as the baseline:

1. create a new folder under `experiments/leduc_poker/`;
2. include a `config.py`, `run.py`, and `README.md`;
3. hold the baseline protocol fixed except for the intended treatment variable;
4. use matched seeds where possible;
5. export the same core metrics and plots so the thesis results have a consistent look and feel.
