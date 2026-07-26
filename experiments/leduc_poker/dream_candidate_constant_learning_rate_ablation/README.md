# DREAM candidate constant learning-rate ablation (Experiment 31)

This experiment varies only the constant optimiser learning rate around the
architecture-selected DREAM candidate. It deliberately does not retest learning
rate decay, because the earlier decay ablation was not helpful.

The default arms are `0.001`, `0.003`, and `0.006`. The baseline is `0.003`.

## Run

```bash
python -m experiments.leduc_poker.dream_candidate_constant_learning_rate_ablation.run
```

## Smoke test

```bash
python -m experiments.leduc_poker.dream_candidate_constant_learning_rate_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants candidate_learning_rate_0_003_baseline,candidate_learning_rate_0_006 \
  --output-root outputs/smoke_tests/dream_candidate_constant_learning_rate_ablation
```

## GCP Batch smoke test

```bash
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp31-constant-lr-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_constant_learning_rate_ablation.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --variants candidate_learning_rate_0_003_baseline,candidate_learning_rate_0_006 \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_constant_learning_rate_ablation" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"
```

Primary plots include `dream_candidate_constant_learning_rate_exploitability_by_nodes.png`.
