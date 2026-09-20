from argparse import Namespace

import torch

from dream_poker.networks import build_network
from dream_poker.solver import DREAMSolver
from experiments.leduc_poker.dream_paper_aligned_frozen_reservoir_audit.config import (
    PAPER_POLICY_FIT_STEPS,
    PRODUCTION_SEEDS,
    TRAINING_CONFIG,
    TRAINING_SECONDS,
    smoke_config,
    task_schedule,
    validate_contract,
)
from gcp.dream_frozen_reservoir_audit_batch import TASK_COUNT, build_job


def _batch_args(kind: str) -> Namespace:
    return Namespace(
        kind=kind,
        repo_url="https://example.invalid/repo.git",
        repo_ref="abc123",
        bucket_root="gs://bucket",
        run_id="drm45-test",
        service_account="batch@example.invalid",
        project_id="project",
        region="europe-west1",
        controller_action="orchestrate",
        parallelism=3,
        module="experiments.leduc_poker.dream_paper_aligned_frozen_reservoir_audit.run",
        controller_runner="gcp/run_dream_paper_aligned_frozen_reservoir_audit.sh",
        experiment_number=45,
        label="dream-paper-aligned",
    )


def test_production_contract_is_paper_aligned_and_paired_with_experiment_44():
    validate_contract(TRAINING_CONFIG, smoke=False)
    assert tuple(TRAINING_CONFIG["seeds"]) == PRODUCTION_SEEDS
    assert task_schedule() == PRODUCTION_SEEDS
    assert TRAINING_CONFIG["training_time_budget_seconds"] == TRAINING_SECONDS == 43_200
    assert TRAINING_CONFIG["num_traversals"] == 900
    assert TRAINING_CONFIG["epsilon"] == 0.50
    assert TRAINING_CONFIG["learning_rate"] == 0.001
    assert TRAINING_CONFIG["batch_size_advantage"] == 2_048
    assert TRAINING_CONFIG["batch_size_baseline"] == 512
    assert TRAINING_CONFIG["batch_size_strategy"] == 2_048
    assert TRAINING_CONFIG["advantage_network_train_steps"] == 3_000
    assert TRAINING_CONFIG["baseline_network_train_steps"] == 1_000
    assert TRAINING_CONFIG["baseline_network_train_every"] == 1
    assert TRAINING_CONFIG["policy_network_train_steps"] == PAPER_POLICY_FIT_STEPS == 4_000
    assert TRAINING_CONFIG["advantage_memory_capacity"] == 2_000_000
    assert TRAINING_CONFIG["baseline_memory_capacity"] == 200_000
    assert TRAINING_CONFIG["strategy_memory_capacity"] == 4_000_000
    assert TRAINING_CONFIG["gradient_clip_norm"] == 1.0
    assert TRAINING_CONFIG["advantage_network_reinitialize_every_iteration"] is True
    assert TRAINING_CONFIG["policy_training_mode"] == "final_only"
    assert TRAINING_CONFIG["policy_network_layers"] == [32, 32]
    assert TRAINING_CONFIG["advantage_network_layers"] == [128, 128]
    assert TRAINING_CONFIG["baseline_network_layers"] == [32, 32]


def test_smoke_contract_keeps_the_treatment_semantics_with_small_budgets():
    config = smoke_config()
    validate_contract(config, smoke=True)
    assert config["num_iterations"] == 3
    assert config["advantage_network_train_steps"] == 1
    assert config["gradient_clip_norm"] == 1.0
    assert config["advantage_network_reinitialize_every_iteration"] is True
    assert config["policy_training_mode"] == "final_only"


def test_cloud_training_uses_three_separate_n2_standard_8_vms():
    job = build_job(_batch_args("train"))
    group = job["taskGroups"][0]
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert TASK_COUNT == 3
    assert group["taskCount"] == 3
    assert group["parallelism"] == 3
    assert group["taskCountPerNode"] == 1
    assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-8"
    assert "dream_paper_aligned_frozen_reservoir_audit" in script
    assert "Experiment 45" in script


def test_advantage_reinitialisation_replaces_network_and_optimizer():
    solver = DREAMSolver.__new__(DREAMSolver)
    solver._advantage_network_type = "mlp"
    solver._info_state_size = 8
    solver._advantage_network_layers = (4, 4)
    solver._num_actions = 3
    solver._current_learning_rate = 0.001
    original = build_network("mlp", 8, (4, 4), 3)
    solver._advantage_networks = [original]
    solver._optimizer_advantages = [torch.optim.Adam(original.parameters(), lr=0.1)]

    solver._reinitialize_advantage_network(0)

    assert solver._advantage_networks[0] is not original
    assert solver._optimizer_advantages[0].param_groups[0]["lr"] == 0.001


def test_gradient_clipping_is_optional_and_enforced_when_configured():
    solver = DREAMSolver.__new__(DREAMSolver)
    solver._gradient_clip_norm = 1.0
    parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    parameter.grad = torch.tensor([3.0, 4.0])

    pre_clip_norm = solver._clip_gradients([parameter])

    assert pre_clip_norm == 5.0
    assert torch.linalg.vector_norm(parameter.grad).item() <= 1.000001
