"""Configuration for the DREAM vectorized-baseline equivalence check."""

import copy
from pathlib import Path

from dream_poker.constants import DEFAULT_SEEDS_5, SMOKE_TEST_SEEDS
from experiments.leduc_poker.dream_candidate_hp_ablation_common import (
    BASE_CANDIDATE_HP_CONFIG,
    make_exp22_candidate_fixed_baseline_config,
    make_variant,
)


EXP22_ORIGINAL_VARIANT = "exp22_candidate_original_learner"
VECTORIZED_BASELINE_VARIANT = "candidate_vectorized_baseline_learner"


IMPLEMENTATION_EQUIVALENCE_VARIANTS = [
    make_variant(
        EXP22_ORIGINAL_VARIANT,
        "Experiment 22 original learner",
        hp_family="baseline_implementation",
        hp_value="original_loop_replay",
        description=(
            "Archived Experiment 22 architecture-selected candidate outputs, "
            "produced with the original per-transition learned-baseline learner."
        ),
        baseline_learner_implementation="original_loop",
        baseline_replay_storage="python_transition_objects",
        compute_baseline_grad_norm_diagnostics=True,
    ),
    make_variant(
        VECTORIZED_BASELINE_VARIANT,
        "Vectorized baseline learner",
        hp_family="baseline_implementation",
        hp_value="vectorized_tensorized_replay",
        description=(
            "Experiment 22 architecture-selected candidate rerun with tensorized "
            "learned-baseline replay, vectorized baseline targets, and optional "
            "baseline gradient-norm diagnostics disabled."
        ),
        baseline_learner_implementation="vectorized_tensorized",
        baseline_replay_storage="tensorized_numpy_arrays",
        compute_baseline_grad_norm_diagnostics=False,
    ),
]


EXPERIMENT_CONFIG = copy.deepcopy(BASE_CANDIDATE_HP_CONFIG)
EXPERIMENT_CONFIG.update(
    {
        "experiment_name": "leduc_poker_dream_vectorized_baseline_equivalence",
        "algorithm": "DREAM-style OpenSpiel vectorized-baseline equivalence check",
        "plot_prefix": "dream_vectorized_baseline_equivalence",
        "plot_title": "DREAM Vectorized-Baseline Equivalence",
        "baseline_variant": EXP22_ORIGINAL_VARIANT,
        "ablation_variants": list(IMPLEMENTATION_EQUIVALENCE_VARIANTS),
        "treatment_keys": [
            "baseline_learner_implementation",
            "baseline_replay_storage",
            "compute_baseline_grad_norm_diagnostics",
        ],
        "fixed_baseline": make_exp22_candidate_fixed_baseline_config(),
        "seeds": list(DEFAULT_SEEDS_5),
        "compute_baseline_grad_norm_diagnostics": False,
        "output_root": Path("outputs") / "dream_vectorized_baseline_equivalence",
    }
)
EXPERIMENT_CONFIG["fixed_baseline"].update(
    {
        "description": (
            "Reuses the Experiment 22 candidate-architecture outputs as the "
            "original-implementation comparator for the vectorized baseline "
            "learner equivalence check."
        ),
    }
)


SMOKE_TEST_CONFIG_OVERRIDES = {
    "seeds": SMOKE_TEST_SEEDS[:1],
    "num_iterations": 3,
    "num_traversals": 4,
    "policy_network_train_steps": 1,
    "advantage_network_train_steps": 1,
    "baseline_network_train_steps": 1,
    "policy_network_train_every": 1,
    "evaluation_interval": 1,
    "output_root": Path("outputs") / "smoke_tests" / "dream_vectorized_baseline_equivalence",
}
