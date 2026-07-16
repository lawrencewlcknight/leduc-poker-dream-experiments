# DREAM sequential versus Ray-parallel equivalence (Experiment 21)

This experiment compares the current best documented DREAM configuration,
uniform `3x64` policy, advantage, and baseline networks, against a Ray-parallel
implementation of the same learner. The parallel arm keeps one central learner
and partitions traversal collection over three Ray workers. Traversal and replay
budgets are partitioned across workers rather than multiplied.

The paired variants are:

- `dream_3x64_sequential`: sequential traversal collection;
- `dream_3x64_ray_parallel`: Ray-parallel traversal collection with three workers.

Both variants run over seeds `1234`, `2025`, and `31415`. The primary quality
check is practical equivalence in final exploitability and final average-policy
value; the primary systems check is whether the parallel arm reduces traversal,
training-loop, and end-to-end wall-clock time.

## Run

```bash
python -m experiments.leduc_poker.dream_parallel_equivalence_ablation.run
```

## Smoke test

```bash
python -m experiments.leduc_poker.dream_parallel_equivalence_ablation.run \
  --seeds 1234 \
  --variant-ids dream_3x64_sequential,dream_3x64_ray_parallel \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --batch-size-advantage 1 \
  --batch-size-strategy 1 \
  --batch-size-baseline 1 \
  --output-root outputs/smoke_tests/dream_parallel_equivalence_ablation
```

## GCP Batch smoke test

```bash
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp21-parallel-equivalence-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_parallel_equivalence_ablation.run \
    --seeds 1234 \
    --variant-ids dream_3x64_sequential,dream_3x64_ray_parallel \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --batch-size-advantage 1 \
    --batch-size-strategy 1 \
    --batch-size-baseline 1 \
    --output-root outputs/cloud/smoke/leduc_dream_parallel_equivalence_ablation" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"
```

Key outputs include `seed_variant_summary.csv`,
`paired_differences_vs_baseline.csv`, `paired_speedups_vs_baseline.csv`,
`paired_equivalence_summary.json`, `paired_speedup_summary.json`, and node-axis
plots under `plots/`.
