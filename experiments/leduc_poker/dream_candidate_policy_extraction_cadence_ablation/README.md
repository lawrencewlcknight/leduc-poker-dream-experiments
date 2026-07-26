# DREAM candidate policy-extraction cadence ablation (Experiment 28)

This experiment varies only `policy_network_train_every` around the
architecture-selected DREAM candidate. It separates the timing of average-policy
training from the number of gradient steps used at each training event.

The default arms train the policy network every `10`, `25`, or `50` DREAM
iterations. The baseline is every `25` iterations.

## Run

```bash
python -m experiments.leduc_poker.dream_candidate_policy_extraction_cadence_ablation.run
```

## Smoke test

```bash
python -m experiments.leduc_poker.dream_candidate_policy_extraction_cadence_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --variants candidate_policy_every_25_baseline,candidate_policy_every_10 \
  --output-root outputs/smoke_tests/dream_candidate_policy_extraction_cadence_ablation
```

## GCP Batch smoke test

```bash
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp28-policy-cadence-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_policy_extraction_cadence_ablation.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --variants candidate_policy_every_25_baseline,candidate_policy_every_10 \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_policy_extraction_cadence_ablation" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"
```

Primary plots include `dream_candidate_policy_extraction_cadence_exploitability_by_nodes.png`.
