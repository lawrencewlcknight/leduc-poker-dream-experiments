import json
from pathlib import Path

from scripts.promote_thesis_artifacts import main


def write_file(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def make_run(run_dir: Path, experiment_name: str = "dream_test_experiment") -> None:
    write_file(
        run_dir / "experiment_metadata.json",
        json.dumps({"experiment_config": {"experiment_name": experiment_name}}),
    )
    write_file(run_dir / "plots" / "curve.png")
    write_file(run_dir / "tables" / "summary.csv")
    write_file(run_dir / "aggregate_summary.json", "{}")
    write_file(run_dir / "paired_difference_summary.json", "{}")
    write_file(run_dir / "paired_aggregate_summary.json", "{}")
    write_file(run_dir / "head_to_head_analysis" / "best_checkpoint_summary.json", "{}")
    write_file(run_dir / "model.pt")
    write_file(run_dir / "checkpoint.pth")
    write_file(run_dir / "arrays.npz")
    write_file(run_dir / "run.log")
    write_file(run_dir / "failed_seeds.json", "{}")
    write_file(run_dir / "failed_runs.json", "{}")
    write_file(run_dir / "checkpoints" / "checkpoint.csv")
    write_file(run_dir / "snapshots" / "snapshot.png")
    write_file(run_dir / "traces" / "trace.csv")


def test_promotes_selected_lightweight_files(tmp_path):
    run_dir = tmp_path / "outputs" / "dream_baseline" / "20260517_120000"
    make_run(run_dir)
    dest = tmp_path / "thesis_artifacts"

    assert main([str(run_dir), "--dest", str(dest)]) == 0

    promoted = dest / "dream_test_experiment" / "20260517_120000"
    assert (promoted / "plots" / "curve.png").exists()
    assert (promoted / "tables" / "summary.csv").exists()
    assert (promoted / "aggregate_summary.json").exists()
    assert (promoted / "paired_difference_summary.json").exists()
    assert (promoted / "paired_aggregate_summary.json").exists()
    assert (promoted / "head_to_head_analysis" / "best_checkpoint_summary.json").exists()
    assert (promoted / "experiment_metadata.json").exists()
    assert (promoted / "promotion_manifest.json").exists()


def test_excludes_heavy_and_scratch_files(tmp_path):
    run_dir = tmp_path / "outputs" / "dream_baseline" / "20260517_120000"
    make_run(run_dir)
    dest = tmp_path / "thesis_artifacts"

    assert main([str(run_dir), "--dest", str(dest)]) == 0

    promoted = dest / "dream_test_experiment" / "20260517_120000"
    assert not (promoted / "model.pt").exists()
    assert not (promoted / "checkpoint.pth").exists()
    assert not (promoted / "arrays.npz").exists()
    assert not (promoted / "run.log").exists()
    assert not (promoted / "failed_seeds.json").exists()
    assert not (promoted / "failed_runs.json").exists()
    assert not (promoted / "checkpoints" / "checkpoint.csv").exists()
    assert not (promoted / "snapshots" / "snapshot.png").exists()
    assert not (promoted / "traces" / "trace.csv").exists()

    manifest = json.loads((promoted / "promotion_manifest.json").read_text(encoding="utf-8"))
    skipped_paths = {item["path"] for item in manifest["skipped_files"]}
    copied_paths = {item["relative_path"] for item in manifest["copied_files"]}
    selected_paths = {item["relative_path"] for item in manifest["selected_files"]}
    assert "model.pt" in skipped_paths
    assert "arrays.npz" in skipped_paths
    assert "failed_runs.json" in skipped_paths
    assert "traces/trace.csv" in skipped_paths
    assert "aggregate_summary.json" in copied_paths
    assert "aggregate_summary.json" in selected_paths


def test_discovers_runs_recursively_from_downloaded_cloud_job(tmp_path):
    job_dir = tmp_path / "cloud_outputs" / "JOB_NAME"
    run_a = job_dir / "outputs" / "cloud" / "experiment_a" / "20260517_120000"
    run_b = job_dir / "outputs" / "cloud" / "experiment_b" / "20260517_130000"
    make_run(run_a, experiment_name="dream_exp_a")
    make_run(run_b, experiment_name="dream_exp_b")
    dest = tmp_path / "thesis_artifacts"

    assert main([str(job_dir), "--dest", str(dest)]) == 0

    assert (dest / "dream_exp_a" / "20260517_120000" / "plots" / "curve.png").exists()
    assert (dest / "dream_exp_b" / "20260517_130000" / "tables" / "summary.csv").exists()
    manifest = json.loads(
        (dest / "dream_exp_a" / "20260517_120000" / "promotion_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["experiment_name"] == "dream_exp_a"
    assert manifest["run_directory_name"] == "20260517_120000"
