
# TODO

- Add future DREAM ablation experiments as separate folders under `experiments/leduc_poker/`, reusing `dream_poker.experiment_runner` where possible.
- Reuse `dream_poker.checkpointing` for future checkpoint or head-to-head stability experiments.
- Reuse `dream_poker.random_search` for future solver-parameter or hyperparameter tuning experiments.
- Reuse `dream_poker.warm_start` for future checkpoint/resume fidelity experiments.
- Reuse `dream_poker.lr_schedule` for future optimiser schedule or decay ablations.
- Reuse `dream_poker.network_budget` for future supervised network-update budget ablations.
- Reuse `dream_poker.variant_ablation` for future matched-seed scalar-parameter ablations.
- Add more detailed unit tests once OpenSpiel is installed in the local development environment.
- Record package versions and hardware metadata for final thesis runs.
