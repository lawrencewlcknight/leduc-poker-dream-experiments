"""Contract, cloud scheduling, and smoke tests for DREAM Experiment 46."""

import json
import subprocess
import sys
from pathlib import Path

from experiments.leduc_poker.dream_paper_aligned_36h_trajectory.cloud import (
    SMOKE_SEEDS,
    aggregate_workers,
    run_evaluation_worker,
    run_training_worker,
    task_name,
)
from experiments.leduc_poker.dream_paper_aligned_36h_trajectory.config import (
    CHECKPOINT_INTERVAL_SECONDS,
    EXPERIMENT_ID,
    PAPER_POLICY_FIT_STEPS,
    PRODUCTION_SEEDS,
    TARGET_NODES_TOUCHED,
    TRAINING_SECONDS,
    build_config,
)


def test_experiment_46_frozen_contract():
    config = build_config()
    assert EXPERIMENT_ID == 46
    assert PRODUCTION_SEEDS == (1234, 2025, 31415, 27182, 16180)
    assert TRAINING_SECONDS == 36 * 60 * 60
    assert CHECKPOINT_INTERVAL_SECONDS == 30 * 60
    assert TARGET_NODES_TOUCHED == 15_000_000
    assert config["num_traversals"] == 900
    assert config["epsilon"] == 0.50
    assert config["advantage_network_train_steps"] == 3_000
    assert config["baseline_network_train_steps"] == 1_000
    assert config["strategy_memory_capacity"] == 4_000_000
    assert config["fresh_reference_fit_steps"] == PAPER_POLICY_FIT_STEPS == 4_000
    assert config["policy_training_mode"] == "final_only"
    assert config["compute_exploitability"] is False
    assert config["exclude_checkpoint_time_from_training_budget"] is True


def test_smoke_contract_reduces_expensive_dimensions():
    config = build_config(smoke=True)
    assert tuple(config["seeds"]) == SMOKE_SEEDS
    assert config["num_traversals"] == 2
    assert config["training_time_budget_seconds"] == 0.01
    assert config["strategy_memory_capacity"] == 256
    assert config["fresh_reference_fit_steps"] == 2


def test_task_schedule_has_one_seed_per_task():
    assert [task_name(index, PRODUCTION_SEEDS) for index in range(5)] == [
        "task_000_seed_1234",
        "task_001_seed_2025",
        "task_002_seed_31415",
        "task_003_seed_27182",
        "task_004_seed_16180",
    ]


def _build_job(tmp_path: Path, kind: str) -> dict:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / f"{kind}.json"
    command = [
        sys.executable,
        str(root / "gcp" / "dream_paper_aligned_36h_trajectory_batch.py"),
        "--kind",
        kind,
        "--output",
        str(output),
        "--run-id",
        "drm46-test",
        "--bucket-root",
        "gs://test-bucket",
        "--service-account",
        "test@example.invalid",
        "--repo-ref",
        "deadbeef",
        "--parallelism",
        "5",
    ]
    if kind == "controller":
        command.extend(["--project-id", "test-project", "--region", "europe-west1"])
    subprocess.run(command, check=True)
    return json.loads(output.read_text(encoding="utf-8"))


def test_training_and_evaluation_use_five_parallel_vms(tmp_path):
    for kind, duration in (("train", "172800s"), ("evaluate", "72000s")):
        job = _build_job(tmp_path, kind)
        group = job["taskGroups"][0]
        assert group["taskCount"] == 5
        assert group["parallelism"] == 5
        assert group["taskCountPerNode"] == 1
        assert group["taskSpec"]["maxRunDuration"] == duration
        assert group["taskSpec"]["maxRetryCount"] == 0
        assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-8"


def test_controller_runs_smoke_training_evaluation_and_aggregation(tmp_path):
    job = _build_job(tmp_path, "controller")
    script = job["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"]
    assert "run_dream_paper_aligned_36h_trajectory.sh" in script
    assert "DRM46_REMOTE_CONTROLLER=1" in script


def test_end_to_end_smoke_produces_temporal_analysis(tmp_path):
    run_training_worker(
        task_index=0,
        seeds=SMOKE_SEEDS,
        output_root=tmp_path,
        smoke=True,
        resume=False,
    )
    run_evaluation_worker(
        task_index=0,
        seeds=SMOKE_SEEDS,
        training_root=tmp_path,
        output_root=tmp_path,
        smoke=True,
        resume=False,
    )
    result = aggregate_workers(
        evaluation_root=tmp_path,
        seeds=SMOKE_SEEDS,
        output_dir=tmp_path / "analysis",
        smoke=True,
    )
    assert result["status"] == "complete"
    assert result["num_seeds"] == 1
    assert (tmp_path / "analysis" / "exploitability_by_training_time.png").is_file()
    assert (tmp_path / "analysis" / "exploitability_by_nodes.png").is_file()
    assert (tmp_path / "analysis" / "distillation_gap_by_training_time.png").is_file()
    assert (tmp_path / "analysis" / "endpoint_aggregate_summary.csv").is_file()
    training_result = json.loads(
        next((tmp_path / "training").rglob("training_result.json")).read_text(
            encoding="utf-8"
        )
    )
    assert training_result["checkpoint_time_excluded_seconds"] > 0.0
    assert training_result["num_checkpoints"] >= 1
