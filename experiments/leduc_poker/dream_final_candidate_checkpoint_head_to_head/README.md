# DREAM final-candidate checkpoint head-to-head (Experiment 43)

## Research question

Does the exploitability improvement observed during the long-horizon DREAM
training run correspond to progressively stronger direct-play performance?

## Design

Each seed trains one uninterrupted instance of the best DREAM configuration
identified in Experiment 38. Lightweight average-policy snapshots are captured
at 20%, 40%, 60%, 80%, and 100% of the 7,500-iteration budget:

```text
iterations:     1500, 3000, 4500, 6000, 7500
expected nodes:   3M,   6M,   9M,  12M,  15M
seeds:          1234, 2025, 31415, 27182, 16180
```

The training configuration is the high-exploration sparse-baseline arm from
Experiment 38: advantage networks `2x128`, policy and learned-baseline networks
`2x32`, `epsilon=0.20`, `160` traversals per player per DREAM iteration,
learned-baseline fitting every `50` iterations, `50` advantage updates per
player per iteration, `50` baseline updates per trained player, average-policy
training every `25` iterations for `100` minibatches, Adam learning rate
`0.003`, batch size `1024`, replay capacity `1e6`, and linear average-strategy
weighting.

The snapshots are saved by a callback inside the standard `DREAMSolver.solve`
loop, so the experiment uses the same baseline-training cadence and vectorized
learned-baseline implementation as the long-horizon run.

## Evaluation and inference

Leduc permits exact policy evaluation. For each seed, every pair of checkpoints
is evaluated in both seat assignments with OpenSpiel. If `A` is the later
checkpoint and `B` the earlier checkpoint, the reported effect is:

```text
0.5 * (value of A as player 0 against B + value of A as player 1 against B)
```

This removes Monte Carlo match noise. The independent training seed is the
statistical unit. The primary estimand is the mean later-versus-earlier exact EV
within each seed, aggregated across seeds. Adjacent-checkpoint and
final-versus-first contrasts are reported separately. Secondary pairwise tests
use exact one-sided sign-flip tests with Holm correction.

## Run

From the repository root:

```bash
# Five-seed default run: train and analyse
python -m experiments.leduc_poker.dream_final_candidate_checkpoint_head_to_head.run

# Train snapshots only
python -m experiments.leduc_poker.dream_final_candidate_checkpoint_head_to_head.run train

# Re-run analysis against existing snapshots
python -m experiments.leduc_poker.dream_final_candidate_checkpoint_head_to_head.run analyse \
  --run-dir outputs/dream_final_candidate_checkpoint_head_to_head/<timestamp>
```

## Smoke test

```bash
python -m experiments.leduc_poker.dream_final_candidate_checkpoint_head_to_head.run \
  --seeds 1234 \
  --iterations 10 \
  --checkpoint-schedule 2,4,6,8,10 \
  --traversals 4 \
  --evaluation-interval 2 \
  --policy-network-train-every 2 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --baseline-network-train-every 2 \
  --policy-network-layers 8,8 \
  --advantage-network-layers 8,8 \
  --baseline-network-layers 8,8 \
  --batch-size-advantage 2 \
  --batch-size-strategy 2 \
  --batch-size-baseline 2 \
  --memory-capacity 256 \
  --output-root outputs/smoke_tests/dream_final_candidate_checkpoint_head_to_head
```

## GCP Batch smoke test

Set `PROJECT_ID`, `REGION`, `BUCKET`, and `SA_EMAIL` first; see
[`docs/GCP_BATCH_EXPERIMENTS.md`](../../../docs/GCP_BATCH_EXPERIMENTS.md).

```bash
./gcp/submit_batch_experiment.sh \
  "ld-exp43-final-h2h-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_final_candidate_checkpoint_head_to_head.run \
    --seeds 1234 \
    --iterations 10 \
    --checkpoint-schedule 2,4,6,8,10 \
    --traversals 4 \
    --evaluation-interval 2 \
    --policy-network-train-every 2 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --baseline-network-train-every 2 \
    --policy-network-layers 8,8 \
    --advantage-network-layers 8,8 \
    --baseline-network-layers 8,8 \
    --batch-size-advantage 2 \
    --batch-size-strategy 2 \
    --batch-size-baseline 2 \
    --memory-capacity 256 \
    --output-root outputs/cloud/smoke/leduc_dream_final_candidate_checkpoint_head_to_head" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"
```

## Principal outputs

| File | Contents |
| --- | --- |
| `training_stage_metrics.csv` | Actual nodes, elapsed time, replay sizes, and policy-fit counts at each snapshot. |
| `policy_training_curves.csv` | Full intermittent policy-training curve emitted by `DREAMSolver.solve`. |
| `head_to_head_analysis/checkpoint_exploitability_metrics.csv` | Exact NashConv/2, policy value, and value error by seed and checkpoint. |
| `head_to_head_analysis/head_to_head_pairwise.csv` | Exact two-seat EV for every ordered checkpoint pair. |
| `head_to_head_analysis/head_to_head_primary_effect_by_seed.csv` | One independent later-versus-earlier summary effect per seed. |
| `head_to_head_analysis/head_to_head_inference_summary.csv` | Primary, adjacent, and final-versus-first estimates with intervals and exact p-values. |
| `head_to_head_analysis/head_to_head_pairwise_inference.csv` | Secondary pair-specific estimates with Holm-adjusted p-values. |
| `head_to_head_analysis/aggregate_summary.json` | Machine-readable statement of the estimands and inference protocol. |
| `head_to_head_analysis/plots/head_to_head_later_vs_earlier_by_nodes.png` | Lower-triangular exact-EV matrix labelled by mean nodes touched. |
| `head_to_head_analysis/plots/head_to_head_strength_vs_earlier_by_nodes.png` | Mean EV against all earlier checkpoints over nodes. |
| `head_to_head_analysis/plots/head_to_head_strength_vs_previous_by_nodes.png` | Adjacent-checkpoint EV over nodes. |
| `head_to_head_analysis/plots/exploitability_by_nodes.png` | Exact exploitability at the five snapshots. |
| `snapshots/leduc_poker_dream_seed_<seed>_policy_snapshot_<iter>_iters.pt` | Lightweight average-policy snapshots used for exact analysis. |
