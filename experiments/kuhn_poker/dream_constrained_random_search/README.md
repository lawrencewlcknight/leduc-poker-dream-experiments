# DREAM constrained random search

This experiment tests whether DREAM performance in Kuhn poker can be improved by tuning solver parameters while keeping the OpenSpiel DREAM-style implementation fixed.

It uses a bounded two-stage design:

- screening evaluates the baseline plus randomly sampled candidate configurations with a shorter budget;
- confirmation reruns the baseline and the best screened candidates with a longer budget and matched seeds.

The default command runs both stages:

```bash
python -m experiments.kuhn_poker.dream_constrained_random_search.run
```

## Smoke test

```bash
python -m experiments.kuhn_poker.dream_constrained_random_search.run \
  --screening-seeds 1234 \
  --confirmation-seeds 1234 \
  --screening-iterations 5 \
  --confirmation-iterations 5 \
  --n-random-candidates 1 \
  --n-confirmation-candidates 1 \
  --num-traversals 20 \
  --policy-network-train-steps 5 \
  --advantage-network-train-steps 5 \
  --baseline-network-train-steps 5 \
  --evaluation-interval 5 \
  --output-root outputs/smoke_tests/dream_constrained_random_search
```

Key outputs are written under `tables/`, `traces/`, and `plots/`. The primary thesis table is `tables/confirmation_config_summary.csv`; the screening stage is exploratory and is used to select candidates for confirmation.
