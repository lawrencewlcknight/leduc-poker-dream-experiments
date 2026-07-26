# DREAM candidate advantage-fitting steps ablation (Experiment 29)

This experiment varies only `advantage_network_train_steps` around the
architecture-selected DREAM candidate. It tests whether the larger `2x128`
advantage networks need more supervised optimisation, or whether fewer updates
regularise noisy sampled regret targets.

The default arms are `25`, `50`, `100`, and `200` advantage-network update steps
per player per DREAM iteration. The baseline is `50`.

## Run

```bash
python -m experiments.leduc_poker.dream_candidate_advantage_fitting_steps_ablation.run
```

## Smoke test

```bash
python -m experiments.leduc_poker.dream_candidate_advantage_fitting_steps_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants candidate_advantage_steps_50_baseline,candidate_advantage_steps_25 \
  --output-root outputs/smoke_tests/dream_candidate_advantage_fitting_steps_ablation
```

## GCP Batch smoke test

```bash
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp29-advantage-steps-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_advantage_fitting_steps_ablation.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --variants candidate_advantage_steps_50_baseline,candidate_advantage_steps_25 \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_advantage_fitting_steps_ablation" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"
```

Primary plots include `dream_candidate_advantage_fitting_steps_exploitability_by_nodes.png`.
