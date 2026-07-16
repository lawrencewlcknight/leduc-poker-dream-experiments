import math

from dream_poker.parallel_utils import equivalence_summary, partition_total, worker_seed


def test_partition_total_preserves_budget_and_balances_workers():
    counts = partition_total(10, 3)
    assert counts == [4, 3, 3]
    assert sum(counts) == 10
    assert max(counts) - min(counts) <= 1


def test_partition_total_supports_more_workers_than_items():
    counts = partition_total(2, 4)
    assert counts == [1, 1, 0, 0]
    assert sum(counts) == 2


def test_worker_seed_is_deterministic_and_distinct():
    seeds = [worker_seed(1234, index) for index in range(3)]
    assert seeds == [1001237, 2001240, 3001243]
    assert len(set(seeds)) == 3


def test_equivalence_summary_marks_small_paired_deltas_equivalent():
    summary = equivalence_summary([0.001, -0.002, 0.0], margin=0.05)
    assert summary["n"] == 3
    assert math.isclose(summary["margin"], 0.05)
    assert summary["all_seed_deltas_within_margin"] is True
    assert summary["tost_equivalent"] is True
