# DREAM vectorized-baseline equivalence check (Experiment 37)

This experiment reruns the Experiment 22 architecture-selected DREAM candidate
using the vectorized learned-baseline implementation. It reuses the archived
Experiment 22 candidate outputs as the original-implementation comparator, so
the default run trains only the new vectorized implementation.

The purpose is to check that tensorized learned-baseline replay, vectorized
baseline-target construction, and disabled baseline grad-norm diagnostics do not
materially change exploitability or average-policy value, while reducing
wall-clock time and cumulative learned-baseline training time.

## Run

```bash
python -m experiments.leduc_poker.dream_vectorized_baseline_equivalence.run
```

## Smoke test

```bash
python -m experiments.leduc_poker.dream_vectorized_baseline_equivalence.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests/dream_vectorized_baseline_equivalence
```

## GCP Batch smoke test

```bash
./gcp/submit_batch_experiment.sh \
  "ld-exp37-vbleq-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_vectorized_baseline_equivalence.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --output-root outputs/cloud/smoke/leduc_dream_vectorized_baseline_equivalence" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"
```

Primary outputs include `paired_differences_vs_original_learner.csv`,
`runtime_summary_by_variant.csv`, and timing plots for final wall-clock seconds,
cumulative learned-baseline training seconds, and cumulative supervised-training
seconds.
