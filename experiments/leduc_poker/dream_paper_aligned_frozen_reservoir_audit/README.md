# Paper-aligned DREAM frozen-reservoir audit (Experiment 45)

## Research question

Does the numerical training regime reported for the original DREAM Leduc
experiments improve the 12-hour performance of the Experiment 44 implementation
when its neural architecture and OpenSpiel representation are held fixed?

## Design

This is a paired follow-on to Experiment 44. Seeds `1234`, `2025`, and `31415`
again train independently for a 12-hour active wall-clock budget on separate
`n2-standard-8` VMs. Relative to Experiment 44, it uses:

- 900 outcome-sampling traversals per player and iteration;
- `epsilon=0.50` and linear CFR weighting;
- 3,000 advantage updates with minibatches of 2,048;
- 1,000 learned-baseline updates every iteration with minibatches of 512;
- two-million-entry advantage reservoirs per player, a 200,000-entry circular
  baseline buffer per player, and a four-million-entry shared strategy reservoir
  (the two-player aggregate equivalent of two million entries per player);
- Adam with learning rate `0.001` and global gradient-norm clipping at `1.0`;
- a freshly initialised advantage network before each player update; and
- fresh 4,000-minibatch average-policy fits after data collection.

The existing `128x128` advantage and `32x32` baseline and policy networks are
held fixed. The strategy reservoir remains shared because this experiment tests
the paper-aligned numerical schedule inside the existing Experiment 44
implementation; it is not presented as a bit-for-bit reproduction of PokerRL.

No average-policy fitting occurs inside the active 12-hour collection budget.
After the reservoir is frozen, the same four policy-fitting arms as Experiment
44 are evaluated: row-wise weighted MSE, row-wise weighted cross-entropy,
grouped weighted cross-entropy, and four-times-extended grouped cross-entropy.
The first three receive 4,000 updates; the extended arm receives 16,000.

## Cloud execution

From the repository root, with the standard DREAM GCP variables set:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="drm45-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=3

./gcp/run_dream_paper_aligned_frozen_reservoir_audit.sh smoke-local
./gcp/run_dream_paper_aligned_frozen_reservoir_audit.sh run
```

Status and recovery:

```bash
./gcp/run_dream_paper_aligned_frozen_reservoir_audit.sh status
./gcp/run_dream_paper_aligned_frozen_reservoir_audit.sh resume
```

The remote controller runs the cloud smoke test, launches the three seed VMs in
parallel, and aggregates their outputs. The laptop can be disconnected after
the controller has been submitted.
