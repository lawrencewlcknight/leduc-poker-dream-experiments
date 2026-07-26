# DREAM candidate advantage batch-size ablation (Experiment 30)

This experiment varies only `batch_size_advantage` around the
architecture-selected DREAM candidate. It tests whether sampled regret fitting
benefits from noisier, potentially regularising minibatches or from larger,
lower-variance batches.

The default arms are advantage minibatches of `512`, `1024`, and `2048`. The
baseline is `1024`.

## Run

```bash
python -m experiments.leduc_poker.dream_candidate_advantage_batch_size_ablation.run
```

## Smoke test

```bash
python -m experiments.leduc_poker.dream_candidate_advantage_batch_size_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants candidate_advantage_batch_1024_baseline,candidate_advantage_batch_512 \
  --output-root outputs/smoke_tests/dream_candidate_advantage_batch_size_ablation
```

## GCP Batch smoke test

```bash
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp30-advantage-batch-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_advantage_batch_size_ablation.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --variants candidate_advantage_batch_1024_baseline,candidate_advantage_batch_512 \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_advantage_batch_size_ablation" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"
```

Primary plots include `dream_candidate_advantage_batch_size_exploitability_by_nodes.png`.
