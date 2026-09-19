# DREAM frozen-reservoir distillation audit (Experiment 44)

## Research question

How much of the selected DREAM candidate's residual Leduc exploitability is
introduced when its sampled average strategy is distilled into the neural
average-policy network?

## Design

The experiment starts from Experiment 43's selected configuration: `2x128`
player-specific advantage networks, `2x32` policy and learned-baseline
networks, `epsilon=0.20`, 160 traversals per player per iteration, learned
baseline fitting every 50 iterations, one-million-entry replay capacities, and
linear iteration weighting with DREAM's reach-ratio correction.

Seeds `1234`, `2025`, and `31415` each train for a 12-hour active wall-clock
budget on a separate `n2-standard-8` VM. The budget is checked between complete
DREAM iterations; setup, reservoir export, frozen-policy fitting, exact
evaluation, and upload are outside the 12-hour training budget.

Each final strategy reservoir is exported as typed compressed arrays. Repeated
rows with the same information-state tensor are collapsed into the exact
sufficient statistic

```text
mass(I)   = sum_j iteration_j * clipped_reach_ratio_j
target(I) = sum_j mass_j * strategy_j / mass(I)
```

The grouped empirical policy is evaluated exactly with OpenSpiel. Four fresh
neural policies then use the same initialization for each seed:

1. row-wise weighted probability MSE;
2. row-wise weighted soft-target cross-entropy;
3. grouped weighted soft-target cross-entropy;
4. grouped weighted soft-target cross-entropy with four times as many updates.

The reference update budget equals the number of policy-gradient steps consumed
by that seed's warm-started Experiment 43 extractor during its realised 12-hour
trajectory. Cross-entropy is masked to legal actions. All arms preserve both
iteration weighting and the sampled-traversal reach correction.

## Cloud execution

The launcher submits a remote controller. It first runs the cloud smoke test,
then starts one three-task Batch array job with `taskCountPerNode=1`, so all
three production seeds run concurrently on separate VMs. Aggregation begins
only after every worker succeeds. Once the controller has been submitted, the
local computer may be disconnected.

From the repository root, with the standard DREAM GCP variables already set:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="drm44-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=3

./gcp/run_dream_frozen_reservoir_audit.sh smoke-local
./gcp/run_dream_frozen_reservoir_audit.sh run
```

Status and recovery:

```bash
./gcp/run_dream_frozen_reservoir_audit.sh status
./gcp/run_dream_frozen_reservoir_audit.sh resume
```

If a worker fails after exporting its reservoir, `resume` downloads that
reservoir and restarts at the frozen distillation stage rather than repeating
the 12-hour DREAM training stage.

## Outputs

Each worker uploads:

```text
workers/task_<index>_seed_<seed>/
  frozen_strategy_reservoir.npz
  grouped_empirical_policy.npz
  training_curves.csv
  training_summary.json
  empirical_policy_metrics.json
  distillation_results.csv
  policies/*.pt
  SUCCESS.json
```

The aggregate stage writes:

```text
analysis/seed_arm_results.csv
analysis/aggregate_summary_by_arm.csv
analysis/paired_differences_vs_row_mse.csv
analysis/training_summaries.csv
analysis/training_summaries.json
analysis/analysis_summary.json
analysis/frozen_reservoir_policy_exploitability.png
analysis/frozen_reservoir_distillation_gap.png
```
