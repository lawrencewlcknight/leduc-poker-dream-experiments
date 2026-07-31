# DREAM candidate epsilon comparison with baseline cadence 50 (Experiment 34)

This experiment tests whether the stronger exploration value remains beneficial
when the learned DREAM baseline is trained sparsely. It starts from the
Experiment 22 architecture-selected candidate and changes only:

- `baseline_network_train_every=50` in both arms;
- `epsilon`, comparing `0.06` against `0.20`.

Both arms are trained over three seeds. Because both arms change the baseline
training cadence away from Experiment 22, the experiment intentionally trains
both variants and does not reuse the fixed Experiment 22 comparator artifact.

## Run

```bash
python -m experiments.leduc_poker.dream_candidate_epsilon_baseline50_comparison.run
```

## Smoke test

```bash
python -m experiments.leduc_poker.dream_candidate_epsilon_baseline50_comparison.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests/dream_candidate_epsilon_baseline50_comparison
```

## GCP Batch smoke test

```bash
./gcp/submit_batch_experiment.sh \
  "ld-exp34-eb50-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_epsilon_baseline50_comparison.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_epsilon_baseline50_comparison" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"
```

Primary plots include
`dream_candidate_epsilon_baseline50_exploitability_by_nodes.png` and
`dream_candidate_epsilon_baseline50_average_policy_value_by_nodes.png`.
