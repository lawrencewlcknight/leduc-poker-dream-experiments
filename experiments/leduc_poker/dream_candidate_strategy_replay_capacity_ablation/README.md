# DREAM candidate strategy replay-capacity ablation (Experiment 25)

This experiment starts from the architecture-selected DREAM candidate and varies
only `strategy_memory_capacity`. It tests whether the average-policy extractor
benefits from fresher strategy replay rather than a large reservoir containing
more early-training behaviour.

The default arms are `1e6`, `5e5`, and `1e5` strategy-memory entries. All other
settings are fixed, including `[32, 32]` policy and baseline networks, `[128,
128]` advantage networks, linear average-strategy weighting, and `epsilon=0.10`.

## Run

```bash
python -m experiments.leduc_poker.dream_candidate_strategy_replay_capacity_ablation.run
```

## Smoke test

```bash
python -m experiments.leduc_poker.dream_candidate_strategy_replay_capacity_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants candidate_strategy_memory_1m_baseline,candidate_strategy_memory_100k \
  --output-root outputs/smoke_tests/dream_candidate_strategy_replay_capacity_ablation
```

## GCP Batch smoke test

```bash
./gcp/submit_batch_experiment.sh \
  "leduc-dream-exp25-strategy-replay-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.dream_candidate_strategy_replay_capacity_ablation.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --baseline-network-train-steps 1 \
    --evaluation-interval 1 \
    --variants candidate_strategy_memory_1m_baseline,candidate_strategy_memory_100k \
    --output-root outputs/cloud/smoke/leduc_dream_candidate_strategy_replay_capacity_ablation" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"
```

Primary plots include `dream_candidate_strategy_replay_capacity_exploitability_by_nodes.png`.
