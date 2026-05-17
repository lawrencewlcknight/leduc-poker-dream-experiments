# Thesis artifact promotion

Full experiment outputs are scratch data. They can be large, noisy, and expensive to keep under version control. The `outputs/` tree and downloaded cloud job folders should therefore be treated as working directories rather than as tracked thesis artifacts.

Curated thesis-facing outputs live under:

```text
thesis_artifacts/<experiment_name>/<run_directory_name>/
```

Only lightweight artifacts should be promoted:

- graph images;
- CSV tables;
- aggregate summary JSON files;
- experiment metadata and provenance.

Heavy or operational artifacts should stay out of git:

- model checkpoints and policy snapshots;
- logs and failed-run tracebacks;
- trace directories;
- NumPy arrays;
- other intermediate scratch files.

## Promote a Completed Job

After downloading a completed cloud job to `cloud_outputs/JOB_NAME`, preview the promotion:

```bash
python scripts/promote_thesis_artifacts.py cloud_outputs/JOB_NAME --dry-run
```

Promote the lightweight artifacts:

```bash
python scripts/promote_thesis_artifacts.py cloud_outputs/JOB_NAME
```

Refresh an existing promoted run:

```bash
python scripts/promote_thesis_artifacts.py cloud_outputs/JOB_NAME --overwrite
```

The script can also be pointed at a specific timestamped run directory or at a local parent directory containing many run directories:

```bash
python scripts/promote_thesis_artifacts.py outputs/dream_multiseed_baseline/20260517_120000
python scripts/promote_thesis_artifacts.py outputs/dream_multiseed_baseline
```

## What Gets Copied

The promotion script discovers run directories by searching for `experiment_metadata.json`.

By default it includes:

```text
*.png
*.csv
aggregate_summary.json
paired_difference_summary.json
paired_aggregate_summary.json
best_checkpoint_summary.json
experiment_metadata.json
```

By default it excludes:

```text
*.pt
*.pth
*.npz
*.log
failed_seeds.json
checkpoints/*
snapshots/*
traces/*
```

Additional include or exclude globs can be supplied when needed:

```bash
python scripts/promote_thesis_artifacts.py cloud_outputs/JOB_NAME \
  --include aggregate_summary_by_variant.json \
  --exclude head_to_head_analysis/raw/*
```

## Promotion Manifest

Each promoted run receives a `promotion_manifest.json` file. It records:

- source and destination paths;
- selected files;
- skipped files;
- promotion timestamp;
- whether `--dry-run` was used;
- whether `--overwrite` was used.

This manifest is intended to make thesis figures and tables auditable without committing the full scratch output tree.

## Why Promotion Is Local

Cloud Batch jobs should not automatically push files back to git. Keeping promotion local has a few advantages:

- you can inspect the run before deciding it belongs in the thesis artifact set;
- failed or partial outputs do not accidentally become tracked files;
- credentials for git pushes are not needed on transient cloud VMs;
- the repo history stays focused on curated, reviewed artifacts rather than every cloud job attempt.

The cloud job should copy outputs to Cloud Storage. After that, download the job outputs locally, inspect them, run the promotion script, review the git diff, and commit only the curated artifacts you want to keep.
