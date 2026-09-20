#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DREAM_AUDIT_EXPERIMENT_NUMBER=45
export DREAM_AUDIT_MODULE="experiments.leduc_poker.dream_paper_aligned_frozen_reservoir_audit.run"
export DREAM_AUDIT_CONTROLLER_RUNNER="gcp/run_dream_paper_aligned_frozen_reservoir_audit.sh"
export DREAM_AUDIT_LABEL="dream-paper-aligned"
export DREAM_AUDIT_RUN_PREFIX="drm45"

exec bash "$SCRIPT_DIR/run_dream_frozen_reservoir_audit.sh" "${1:-run}"
