
# Testing

This file records the basic checks used for the DREAM Kuhn poker experiments repository.

## Environment setup

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

OpenSpiel installation can vary by platform. Follow the official OpenSpiel installation instructions if the wheel is unavailable for your system.

## Syntax/import checks

```bash
python -m compileall dream_poker experiments tests
pytest -q
```

## Quick smoke test

The following command runs a deliberately tiny two-seed DREAM experiment. It is intended to check that the runner, outputs, and plotting pipeline work; it is not scientifically meaningful.

```bash
python -m experiments.kuhn_poker.dream_multiseed_baseline.run       --seeds 1234,2025       --iterations 10       --traversals 50       --policy-network-train-steps 20       --advantage-network-train-steps 20       --baseline-network-train-steps 20       --evaluation-interval 5       --output-root outputs/smoke_tests/dream_multiseed_baseline
```

## Full baseline run

```bash
python -m experiments.kuhn_poker.dream_multiseed_baseline.run
```

The full run can be computationally expensive. Use the smoke test first after making code changes.
