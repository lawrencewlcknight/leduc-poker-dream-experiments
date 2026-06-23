# Running the DREAM experiments on Google Cloud Batch

This guide explains how to run the DREAM Leduc poker experiments on Google Cloud using Google Batch. The workflow is designed to be repeatable and command-line driven:

1. configure Google Cloud locally;
2. create a Cloud Storage bucket for outputs;
3. create a service account for Batch jobs;
4. create the Batch submission script;
5. run a smoke test;
6. run full DREAM experiments with configurable CPU and memory;
7. inspect logs and retrieve outputs.

Each Batch job creates a temporary VM, clones this repository, creates an isolated Python 3.9 virtual environment, installs the repository dependencies, runs the selected experiment, copies outputs to Cloud Storage, and exits. Batch handles VM lifecycle management, so there is no persistent VM to shut down after a successful job.

This repository pins Python 3.9 in:

- `.python-version`
- `runtime.txt`
- `pyproject.toml`

The Batch script below also creates its virtual environment with `python3.9` explicitly.

---

## 1. Prerequisites

You need:

- a Google Cloud project with billing enabled;
- the Google Cloud CLI installed on your local machine;
- permission to create service accounts, IAM bindings, Batch jobs, and Cloud Storage buckets;
- this GitHub repository available to the Batch VM.

If the repository is public, the script can clone it directly with HTTPS. If the repository is private, adapt the `git clone` step to use an authenticated method such as a deploy key, GitHub token, or a pre-built container image.

---

## 2. One-time local Google Cloud setup

Authenticate and select your project:

```bash
gcloud init
gcloud auth login

export PROJECT_ID="your-gcp-project-id"
gcloud config set project "$PROJECT_ID"
```

For UK-based use where latency is not important and cost is a consideration, `europe-west1` is a sensible default region:

```bash
export REGION="europe-west1"
export ZONE="europe-west1-b"
```

Enable the required APIs:

```bash
gcloud services enable \
  compute.googleapis.com \
  batch.googleapis.com \
  logging.googleapis.com \
  storage.googleapis.com
```

---

## 3. Create a Cloud Storage bucket for experiment outputs

Create a regional bucket in the same region as the Batch jobs:

```bash
export BUCKET_NAME="${PROJECT_ID}-leduc-poker-dream-results"
export BUCKET="gs://${BUCKET_NAME}"

gcloud storage buckets create "$BUCKET" \
  --location="$REGION" \
  --uniform-bucket-level-access
```

Check the bucket exists:

```bash
gcloud storage buckets describe "$BUCKET"
```

If the bucket exists and is accessible, this command will print metadata such as the bucket name, creation time, location, storage class, and storage URL.

---

## 4. Create a service account for Batch jobs

Create a dedicated service account:

```bash
export SA_NAME="leduc-dream-runner"
export SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts create "$SA_NAME" \
  --display-name="Leduc poker DREAM experiment runner" \
  --project="$PROJECT_ID"
```

Grant the service account permission to write logs:

```bash
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/logging.logWriter"
```

Grant the service account permission to report Batch agent status:

```bash
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/batch.agentReporter"
```

Grant the service account permission to write experiment outputs to the bucket:

```bash
gcloud storage buckets add-iam-policy-binding "$BUCKET" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.objectAdmin"
```

Allow your user account to run jobs as this service account. Replace the email address with your Google account email:

```bash
export YOUR_EMAIL="your-email@example.com"

gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --member="user:${YOUR_EMAIL}" \
  --role="roles/iam.serviceAccountUser"
```

If you need to inspect logs from your local account, make sure your user has log-viewing permission:

```bash
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="user:${YOUR_EMAIL}" \
  --role="roles/logging.viewer"
```

---

## 5. Environment variables to set in each new terminal

Before submitting jobs from a new shell session, set:

```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="europe-west1"
export BUCKET="gs://${PROJECT_ID}-leduc-poker-dream-results"
export SA_EMAIL="leduc-dream-runner@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud config set project "$PROJECT_ID"
```

Check the values:

```bash
echo "$PROJECT_ID"
echo "$REGION"
echo "$BUCKET"
echo "$SA_EMAIL"
```

---

## 6. Create the Batch submission script

Create a `gcp` directory in the repository root:

```bash
mkdir -p gcp
```

Create the script:

```bash
nano gcp/submit_batch_experiment.sh
```

Paste the following content into the file:

```bash
#!/usr/bin/env bash
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive

# Usage:
#   ./gcp/submit_batch_experiment.sh \
#     JOB_NAME \
#     "PYTHON_EXPERIMENT_COMMAND" \
#     MACHINE_TYPE \
#     MAX_RUN_SECONDS \
#     CPU_MILLI \
#     MEMORY_MIB
#
# Examples:
#   n2-standard-2: CPU_MILLI=2000 MEMORY_MIB=8000
#   n2-standard-4: CPU_MILLI=4000 MEMORY_MIB=16000
#   n2-standard-8: CPU_MILLI=8000 MEMORY_MIB=32000

JOB_NAME="$1"
EXPERIMENT_COMMAND="$2"
MACHINE_TYPE="${3:-n2-standard-4}"
MAX_RUN_SECONDS="${4:-21600}"
CPU_MILLI="${5:-4000}"
MEMORY_MIB="${6:-16000}"

: "${PROJECT_ID:?Set PROJECT_ID first}"
: "${REGION:?Set REGION first}"
: "${BUCKET:?Set BUCKET first}"
: "${SA_EMAIL:?Set SA_EMAIL first}"

JOB_JSON="$(mktemp "/tmp/${JOB_NAME}.XXXXXX.json")"

export JOB_NAME
export EXPERIMENT_COMMAND
export MACHINE_TYPE
export MAX_RUN_SECONDS
export CPU_MILLI
export MEMORY_MIB
export BUCKET
export SA_EMAIL
export JOB_JSON

python3 <<'PY'
import json
import os

job_json_path = os.environ["JOB_JSON"]
job_name = os.environ["JOB_NAME"]
experiment_command = os.environ["EXPERIMENT_COMMAND"]
machine_type = os.environ["MACHINE_TYPE"]
max_run_seconds = os.environ["MAX_RUN_SECONDS"]
cpu_milli = int(os.environ["CPU_MILLI"])
memory_mib = int(os.environ["MEMORY_MIB"])
bucket = os.environ["BUCKET"]
service_account = os.environ["SA_EMAIL"]

script = f"""#!/usr/bin/env bash
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive

echo "Starting job: {job_name}"
echo "Experiment command: {experiment_command}"
echo "Requested CPU milli: {cpu_milli}"
echo "Requested memory MiB: {memory_mib}"

if command -v sudo >/dev/null 2>&1; then
  SUDO=sudo
else
  SUDO=
fi

$SUDO apt-get update
$SUDO apt-get install -y git python3.9 python3.9-dev python3.9-venv

WORKDIR=/workspace
mkdir -p "$WORKDIR"
cd "$WORKDIR"

git clone --depth 1 https://github.com/lawrencewlcknight/leduc-poker-dream-experiments.git
cd leduc-poker-dream-experiments

export HOME="${{HOME:-/root}}"
export TMPDIR="/tmp"
export PIP_CACHE_DIR="/tmp/pip-cache"
export MPLCONFIGDIR="/tmp/matplotlib-cache"
export PATH="/usr/local/bin:$PATH"

mkdir -p "$HOME" "$TMPDIR" "$PIP_CACHE_DIR" "$MPLCONFIGDIR"

echo "Python version:"
python3.9 --version

echo "Machine information:"
nproc || true
free -h || true
df -h || true
lscpu | head -30 || true

# Keep experiment dependencies isolated from the Google Cloud CLI Python runtime.
python3.9 -m venv --copies /tmp/leduc-dream-venv
source /tmp/leduc-dream-venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir --no-build-isolation -r requirements.txt
# Skipping editable install: this repo is run from the repository root.
python -m pip check || true

mkdir -p "outputs/cloud/{job_name}"

{experiment_command}

deactivate

echo "Experiment completed. Copying outputs to Cloud Storage."
gsutil -m cp -r outputs "{bucket}/{job_name}/"

echo "Done."
"""

job = {
    "taskGroups": [
        {
            "taskSpec": {
                "runnables": [
                    {
                        "script": {
                            "text": script
                        }
                    }
                ],
                "computeResource": {
                    "cpuMilli": cpu_milli,
                    "memoryMib": memory_mib,
                },
                "maxRetryCount": 0,
                "maxRunDuration": f"{max_run_seconds}s",
            },
            "taskCount": 1,
            "parallelism": 1,
        }
    ],
    "allocationPolicy": {
        "serviceAccount": {
            "email": service_account
        },
        "instances": [
            {
                "policy": {
                    "machineType": machine_type,
                    "provisioningModel": "STANDARD",
                }
            }
        ],
    },
    "logsPolicy": {
        "destination": "CLOUD_LOGGING"
    },
}

with open(job_json_path, "w", encoding="utf-8") as f:
    json.dump(job, f, indent=2)
PY

echo "Submitting Batch job: ${JOB_NAME}"
echo "Machine type: ${MACHINE_TYPE}"
echo "Max run duration: ${MAX_RUN_SECONDS}s"
echo "CPU milli: ${CPU_MILLI}"
echo "Memory MiB: ${MEMORY_MIB}"
echo "Job config: ${JOB_JSON}"

echo
echo "Script that will run inside Batch:"
echo "-----------------------------------"
python3 - "$JOB_JSON" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    job = json.load(f)

print(job["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"])
PY
echo "-----------------------------------"
echo

gcloud batch jobs submit "${JOB_NAME}" \
  --location "${REGION}" \
  --config "${JOB_JSON}"

echo "Submitted."
echo "Monitor with:"
echo "  gcloud batch jobs describe ${JOB_NAME} --location ${REGION}"
echo "Outputs will be copied to:"
echo "  ${BUCKET}/${JOB_NAME}/"
```

Make the script executable and check its syntax:

```bash
chmod +x gcp/submit_batch_experiment.sh
bash -n gcp/submit_batch_experiment.sh
```

`bash -n` should return silently. If it prints an error, fix the script before submitting a Batch job.

---

## 7. Run a smoke test

Before running a full experiment, submit a small DREAM baseline smoke test:

```bash
./gcp/submit_batch_experiment.sh \
  "dream-smoke-baseline-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_multiseed_baseline.run \
    --seeds 1234 \
    --iterations 10 \
    --traversals 50 \
    --policy-network-train-steps 20 \
    --advantage-network-train-steps 20 \
    --baseline-network-train-steps 20 \
    --evaluation-interval 5 \
    --output-root outputs/cloud/dream-smoke-baseline" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"
```

The script prints the Batch script before submission. Check that:

- it clones `leduc-poker-dream-experiments`;
- it creates the virtual environment with `python3.9`;
- the upload line uses `gsutil`, for example:

```bash
gsutil -m cp -r outputs "gs://your-project-id-leduc-poker-dream-results/dream-smoke-baseline-.../"
```

After submitting the smoke test, this command can be used to see whether the experiment is queued or running:

```bash
gcloud batch jobs list --location "$REGION"
```

---

## 8. Monitor a Batch job

List jobs:

```bash
gcloud batch jobs list --location "$REGION"
```

Describe one job:

```bash
gcloud batch jobs describe JOB_NAME --location "$REGION"
```

After an experiment has completed, the output from `gcloud batch jobs describe JOB_NAME --location "$REGION"` can also be used to determine how long the Batch job took to run. Review the job status events and timestamp fields in the describe output to compare the start and completion times.

Possible states include:

- `QUEUED`
- `SCHEDULED`
- `RUNNING`
- `SUCCEEDED`
- `FAILED`

If the job succeeds, list the uploaded outputs:

```bash
gcloud storage ls -r "$BUCKET/JOB_NAME/"
```

Copy outputs back to your local machine:

```bash
mkdir -p cloud_outputs/JOB_NAME
gcloud storage cp -r "$BUCKET/JOB_NAME/*" "cloud_outputs/JOB_NAME/"
```

After reviewing the downloaded outputs, promote only lightweight thesis-facing artifacts into the tracked repo:

```bash
python scripts/promote_thesis_artifacts.py cloud_outputs/JOB_NAME --dry-run
python scripts/promote_thesis_artifacts.py cloud_outputs/JOB_NAME
```

See `docs/THESIS_ARTIFACTS.md` for the full local promotion workflow. Full cloud outputs should remain scratch data unless specific lightweight artifacts are intentionally promoted.

---

## 9. Read logs for a failed job

First describe the job and find its `uid`:

```bash
gcloud batch jobs describe JOB_NAME --location "$REGION"
```

Then query task logs using the job UID:

```bash
gcloud logging read \
  'logName="projects/YOUR_PROJECT_ID/logs/batch_task_logs" AND labels.job_uid="JOB_UID"' \
  --limit=500 \
  --format="value(timestamp,severity,textPayload,jsonPayload.message)"
```

Useful things to look for:

- `Python version:` should print Python 3.9;
- `Outputs written to` means the experiment runner completed;
- `Experiment completed. Copying outputs to Cloud Storage.` means the upload step started;
- `No space left on device` indicates disk pressure;
- `Killed`, `exit code 137`, or `Out of memory` usually indicates memory pressure;
- `maxRunDuration` means the job hit the time limit;
- `Invalid machine type` or resource errors usually mean the requested CPU/memory does not fit the selected machine type;
- `Unable to locate package python3.9` means the Batch VM image does not provide Python 3.9 through `apt`; use a VM image with Python 3.9 available or switch to a pre-built container image.

---

## 10. Run full DREAM experiments

After the smoke test succeeds, run the full experiments using their module entry points.

A safe starting configuration is `n2-standard-4`, 4 vCPUs, 16 GiB memory, and a 12-hour runtime cap:

```bash
./gcp/submit_batch_experiment.sh \
  "dream-baseline-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_multiseed_baseline.run \
    --output-root outputs/cloud/dream-baseline" \
  "n2-standard-4" \
  "43200" \
  "4000" \
  "16000"
```

To collect process-level runtime and memory diagnostics, wrap the experiment command with `/usr/bin/time -v`:

```bash
./gcp/submit_batch_experiment.sh \
  "dream-baseline-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.dream_multiseed_baseline.run \
    --output-root outputs/cloud/dream-baseline" \
  "n2-standard-4" \
  "43200" \
  "4000" \
  "16000"
```

In the logs, look for:

- `Elapsed (wall clock) time`;
- `Percent of CPU this job got`;
- `Maximum resident set size`.

These help determine whether the VM is over- or under-sized.

---

## 11. DREAM experiment entry points

Use these module commands as the `PYTHON_EXPERIMENT_COMMAND` argument:

| Experiment | Command |
|---|---|
| Multi-seed baseline | `python -m experiments.leduc_poker.dream_multiseed_baseline.run --output-root outputs/cloud/dream_multiseed_baseline` |
| Final-only average-policy training ablation | `python -m experiments.leduc_poker.dream_final_only_policy_training_ablation.run --output-root outputs/cloud/dream_final_only_policy_training_ablation` |
| Checkpoint stability | `python -m experiments.leduc_poker.dream_checkpoint_stability.run --output-root outputs/cloud/dream_checkpoint_stability` |
| Constrained random search | `python -m experiments.leduc_poker.dream_constrained_random_search.run --output-root outputs/cloud/dream_constrained_random_search` |
| Warm-start ablation | `python -m experiments.leduc_poker.dream_warm_start_ablation.run --output-root outputs/cloud/dream_warm_start_ablation` |
| Learning-rate schedule ablation | `python -m experiments.leduc_poker.dream_lr_schedule_ablation.run --output-root outputs/cloud/dream_lr_schedule_ablation` |
| Baseline-network budget ablation | `python -m experiments.leduc_poker.dream_baseline_network_budget_ablation.run --output-root outputs/cloud/dream_baseline_network_budget_ablation` |
| Epsilon-exploration ablation | `python -m experiments.leduc_poker.dream_epsilon_exploration_ablation.run --output-root outputs/cloud/dream_epsilon_exploration_ablation` |
| Trajectories-per-iteration ablation | `python -m experiments.leduc_poker.dream_trajectories_per_iteration_ablation.run --output-root outputs/cloud/dream_trajectories_per_iteration_ablation` |
| Experiment 10: network-width ablation | `python -m experiments.leduc_poker.dream_network_size_ablation.run --output-root outputs/cloud/dream_network_width_ablation` |
| Experiment 11: network-depth ablation | `python -m experiments.leduc_poker.dream_network_depth_ablation.run --output-root outputs/cloud/dream_network_depth_ablation` |
| Experiment 12: capacity-extremes ablation | `python -m experiments.leduc_poker.dream_network_capacity_extremes_ablation.run --output-root outputs/cloud/dream_network_capacity_extremes_ablation` |

Example:

```bash
./gcp/submit_batch_experiment.sh \
  "dream-lr-schedule-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_lr_schedule_ablation.run \
    --output-root outputs/cloud/dream_lr_schedule_ablation" \
  "n2-standard-4" \
  "43200" \
  "4000" \
  "16000"
```

The network-size study is split into three independent experiments. Submit each
module as its own job:

```bash
./gcp/submit_batch_experiment.sh \
  "dream-exp10-width-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_network_size_ablation.run \
    --output-root outputs/cloud/dream_network_width_ablation" \
  "n2-standard-4" \
  "172800" \
  "4000" \
  "16000"

./gcp/submit_batch_experiment.sh \
  "dream-exp11-depth-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_network_depth_ablation.run \
    --output-root outputs/cloud/dream_network_depth_ablation" \
  "n2-standard-4" \
  "172800" \
  "4000" \
  "16000"

./gcp/submit_batch_experiment.sh \
  "dream-exp12-capacity-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_network_capacity_extremes_ablation.run \
    --output-root outputs/cloud/dream_network_capacity_extremes_ablation" \
  "n2-standard-4" \
  "172800" \
  "4000" \
  "16000"
```

---

## 12. Changing CPU and memory

The final three arguments control the VM and Batch task resources:

```bash
MACHINE_TYPE MAX_RUN_SECONDS CPU_MILLI MEMORY_MIB
```

Examples:

| Machine type | CPU milli | Memory MiB | Approximate resources |
|---|---:|---:|---|
| `n2-standard-2` | `2000` | `8000` | 2 vCPUs, about 8 GiB RAM |
| `n2-standard-4` | `4000` | `16000` | 4 vCPUs, about 16 GiB RAM |
| `n2-standard-8` | `8000` | `32000` | 8 vCPUs, about 32 GiB RAM |

The Batch resource request must fit inside the selected machine type. For example, do not request `4000` CPU milli and `16000` MiB memory on an `n2-standard-2` machine.

### Smaller VM test

```bash
./gcp/submit_batch_experiment.sh \
  "dream-baseline-small-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_multiseed_baseline.run \
    --output-root outputs/cloud/dream-baseline-small" \
  "n2-standard-2" \
  "43200" \
  "2000" \
  "8000"
```

### Larger VM test

```bash
./gcp/submit_batch_experiment.sh \
  "dream-baseline-large-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_multiseed_baseline.run \
    --output-root outputs/cloud/dream-baseline-large" \
  "n2-standard-8" \
  "43200" \
  "8000" \
  "32000"
```

---

## 13. Choosing a VM size

Use evidence rather than guessing.

Start with:

```text
n2-standard-4, CPU_MILLI=4000, MEMORY_MIB=16000
```

Then compare with:

```text
n2-standard-2, CPU_MILLI=2000, MEMORY_MIB=8000
n2-standard-8, CPU_MILLI=8000, MEMORY_MIB=32000
```

A VM may be too large if:

- memory usage is far below the allocation;
- CPU utilisation is consistently low;
- doubling vCPUs does not materially reduce runtime;
- the job is bottlenecked by Python/OpenSpiel traversal rather than CPU capacity.

A VM may be too small if:

- the job fails with `Killed`, `Out of memory`, or exit code `137`;
- the job hits `maxRunDuration`;
- logs show `No space left on device`;
- the job is much slower than expected.

---

## 14. Runtime limits

`MAX_RUN_SECONDS` is a safety cap. For example:

| Seconds | Duration |
|---:|---:|
| `3600` | 1 hour |
| `21600` | 6 hours |
| `43200` | 12 hours |
| `86400` | 24 hours |

If a job exceeds `MAX_RUN_SECONDS`, Batch stops the task and marks the job as failed. If the task is stopped before the final `gsutil` upload step, outputs that exist only on the VM may be lost.

---

## 15. Output layout

Each experiment writes to a timestamped directory under the selected `--output-root`. The Batch script uploads the full local `outputs/` tree to:

```text
gs://<bucket>/<job-name>/outputs/
```

The experiment-specific output conventions are documented in:

```text
docs/OUTPUT_CONVENTIONS.md
```
