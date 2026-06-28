#!/usr/bin/env bash
set -euo pipefail

# Submit GCP Batch smoke tests for the newest DREAM ablations:
#   Experiment 13: advantage-target processing
#   Experiment 14: residual networks
#   Experiment 15: average-strategy weighting
#
# Required environment variables are inherited from submit_batch_experiment.sh:
#   PROJECT_ID, REGION, BUCKET, SA_EMAIL
#
# Usage:
#   ./gcp/submit_recent_ablation_smoke_tests.sh [MACHINE_TYPE] [MAX_RUN_SECONDS] [CPU_MILLI] [MEMORY_MIB]
#
# Set DRY_RUN=1 to print the three submit commands without calling gcloud.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_SCRIPT="${SCRIPT_DIR}/submit_batch_experiment.sh"

MACHINE_TYPE="${1:-n2-standard-4}"
MAX_RUN_SECONDS="${2:-3600}"
CPU_MILLI="${3:-4000}"
MEMORY_MIB="${4:-16000}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

submit_smoke_test() {
  local job_name="$1"
  local experiment_command="$2"

  echo "Submitting ${job_name}"
  if [ "${DRY_RUN}" = "1" ]; then
    printf '%q ' \
      "${SUBMIT_SCRIPT}" \
      "${job_name}" \
      "${experiment_command}" \
      "${MACHINE_TYPE}" \
      "${MAX_RUN_SECONDS}" \
      "${CPU_MILLI}" \
      "${MEMORY_MIB}"
    printf '\n'
    return 0
  fi

  "${SUBMIT_SCRIPT}" \
    "${job_name}" \
    "${experiment_command}" \
    "${MACHINE_TYPE}" \
    "${MAX_RUN_SECONDS}" \
    "${CPU_MILLI}" \
    "${MEMORY_MIB}"
}

submit_smoke_test \
  "leduc-dream-exp13-target-processing-smoke-${RUN_ID}" \
  "python -m experiments.leduc_poker.dream_target_processing_ablation.run --seeds 1234 --iterations 3 --traversals 4 --evaluation-interval 1 --policy-network-train-steps 1 --advantage-network-train-steps 1 --baseline-network-train-steps 1 --batch-size-advantage 1 --batch-size-strategy 1 --batch-size-baseline 1 --variants raw_targets_dream_baseline,standardized_clipped_targets --output-root outputs/cloud/smoke/leduc_dream_target_processing_ablation"

submit_smoke_test \
  "leduc-dream-exp14-residual-smoke-${RUN_ID}" \
  "python -m experiments.leduc_poker.dream_residual_network_ablation.run --seeds 1234 --iterations 3 --traversals 4 --evaluation-interval 1 --policy-network-train-steps 1 --advantage-network-train-steps 1 --baseline-network-train-steps 1 --variants plain_layers2_width32,residual_layers2_width32 --output-root outputs/cloud/smoke/leduc_dream_residual_network_ablation"

submit_smoke_test \
  "leduc-dream-exp15-avg-weighting-smoke-${RUN_ID}" \
  "python -m experiments.leduc_poker.dream_average_strategy_weighting_ablation.run --seeds 1234 --iterations 3 --traversals 4 --evaluation-interval 1 --policy-network-train-steps 1 --advantage-network-train-steps 1 --baseline-network-train-steps 1 --batch-size-advantage 1 --batch-size-strategy 1 --batch-size-baseline 1 --output-root outputs/cloud/smoke/leduc_dream_average_strategy_weighting_ablation"
