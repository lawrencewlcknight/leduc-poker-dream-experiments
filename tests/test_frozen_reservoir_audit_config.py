from argparse import Namespace

from experiments.leduc_poker.dream_frozen_reservoir_distillation_audit.config import (
    ARM_ORDER,
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
        run_id="drm44-test",
        service_account="batch@example.invalid",
        project_id="project",
        region="europe-west1",
        controller_action="orchestrate",
        parallelism=3,
        module="experiments.leduc_poker.dream_frozen_reservoir_distillation_audit.run",
        controller_runner="gcp/run_dream_frozen_reservoir_audit.sh",
        experiment_number=44,
        label="dream-frozen-reservoir",
    )


def test_production_contract_inherits_selected_experiment_43_configuration():
    validate_contract(TRAINING_CONFIG, smoke=False)
    assert tuple(TRAINING_CONFIG["seeds"]) == PRODUCTION_SEEDS
    assert TRAINING_CONFIG["training_time_budget_seconds"] == TRAINING_SECONDS == 43_200
    assert TRAINING_CONFIG["epsilon"] == 0.20
    assert TRAINING_CONFIG["baseline_network_train_every"] == 50
    assert TRAINING_CONFIG["policy_network_layers"] == [32, 32]
    assert TRAINING_CONFIG["advantage_network_layers"] == [128, 128]
    assert TRAINING_CONFIG["strategy_memory_capacity"] == 1_000_000
    assert len(ARM_ORDER) == 4
    assert task_schedule() == PRODUCTION_SEEDS


def test_smoke_contract_is_small_but_complete():
    config = smoke_config()
    validate_contract(config, smoke=True)
    assert config["num_iterations"] == 8
    assert config["strategy_memory_capacity"] == 256
    assert config["smoke_reference_fit_steps"] == 2


def test_cloud_training_uses_three_tasks_on_separate_nodes():
    job = build_job(_batch_args("train"))
    group = job["taskGroups"][0]
    assert TASK_COUNT == 3
    assert group["taskCount"] == 3
    assert group["parallelism"] == 3
    assert group["taskCountPerNode"] == 1
    assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-8"
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert "BATCH_TASK_INDEX" in script
    assert "frozen_strategy_reservoir.npz" not in script or "rsync" in script
