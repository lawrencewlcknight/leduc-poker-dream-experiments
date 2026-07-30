# DREAM candidate baseline-training cadence ablation (Experiment 32)

This experiment tests whether the learned DREAM Q-baseline needs to be trained
on every iteration. It starts from the Experiment 22 architecture-selected
candidate and varies only `baseline_network_train_every`.

The comparator arm is reused from the tracked Experiment 22 candidate artifact,
so the unchanged `baseline_network_train_every=1` arm is not retrained by
default. Treatment arms train the learned baseline on iteration 1 and then every
iteration divisible by `5`, `10`, `25`, or `50`.

## Run

```bash
python -m experiments.leduc_poker.dream_candidate_baseline_training_cadence_ablation.run
```

Pass `--train-baseline` only when intentionally regenerating the Experiment 22
comparator arm for debugging.

## Smoke test

```bash
python -m experiments.leduc_poker.dream_candidate_baseline_training_cadence_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants candidate_baseline_every_1_baseline,candidate_baseline_every_5 \
  --train-baseline \
  --output-root outputs/smoke_tests/dream_candidate_baseline_training_cadence_ablation
```

## GCP Batch smoke test

```bash
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp32-baseline-cadence-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_baseline_training_cadence_ablation.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --variants candidate_baseline_every_1_baseline,candidate_baseline_every_5 \
    --train-baseline \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_baseline_training_cadence_ablation" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"
```

Primary plots include
`dream_candidate_baseline_training_cadence_exploitability_by_nodes.png` and
`dream_candidate_baseline_training_cadence_average_policy_value_by_nodes.png`.
