# DREAM candidate policy-extraction budget ablation (Experiment 27)

This experiment varies only `policy_network_train_steps` around the
architecture-selected DREAM candidate. It tests whether final exploitability is
limited by fitting the evaluated neural average policy rather than by regret
estimation alone.

The default arms are `50`, `100`, `200`, and `400` average-policy update steps
per training event. The baseline is `100`.

## Run

```bash
python -m experiments.leduc_poker.dream_candidate_policy_extraction_budget_ablation.run
```

Full runs reuse the tracked Experiment 22 candidate baseline comparator by default. Add `--train-baseline` only when you intentionally want to retrain that comparator.

## Smoke test

```bash
python -m experiments.leduc_poker.dream_candidate_policy_extraction_budget_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants candidate_policy_steps_100_baseline,candidate_policy_steps_200 \
  --train-baseline \
  --output-root outputs/smoke_tests/dream_candidate_policy_extraction_budget_ablation
```

## GCP Batch smoke test

```bash
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp27-policy-budget-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_policy_extraction_budget_ablation.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --variants candidate_policy_steps_100_baseline,candidate_policy_steps_200 \
    --train-baseline \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_policy_extraction_budget_ablation" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"
```

Primary plots include `dream_candidate_policy_extraction_budget_exploitability_by_nodes.png`.
