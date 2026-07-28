# DREAM candidate traversal-budget ablation (Experiment 24)

This experiment starts from the architecture-selected DREAM candidate and varies
only `num_traversals`, testing whether the improvement seen in random candidate
03 was driven by lower outcome-sampling noise. The default arms are `160`,
`320`, and `480` traversals per player per DREAM iteration.

The baseline uses policy and learned-baseline networks `[32, 32]`, advantage
networks `[128, 128]`, linear average-strategy weighting, `epsilon=0.06`, and
three matched seeds.

## Run

```bash
python -m experiments.leduc_poker.dream_candidate_traversal_budget_ablation.run
```

Full runs reuse the tracked Experiment 22 candidate baseline comparator by default. Add `--train-baseline` only when you intentionally want to retrain that comparator.

## Smoke test

```bash
python -m experiments.leduc_poker.dream_candidate_traversal_budget_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants candidate_traversals_160_baseline,candidate_traversals_320 \
  --train-baseline \
  --output-root outputs/smoke_tests/dream_candidate_traversal_budget_ablation
```

## GCP Batch smoke test

```bash
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp24-traversal-budget-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_traversal_budget_ablation.run \
    --seeds 1234 \
    --iterations 3 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --variants candidate_traversals_160_baseline,candidate_traversals_320 \
    --train-baseline \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_traversal_budget_ablation" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"
```

Primary plots include `dream_candidate_traversal_budget_exploitability_by_nodes.png`.
