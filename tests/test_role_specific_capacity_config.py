from experiments.leduc_poker.dream_role_specific_capacity_ablation.config import (
    BASELINE_VARIANT,
    ROLE_SPECIFIC_CAPACITY_VARIANTS,
)


def variants_by_id():
    return {variant["variant_id"]: variant for variant in ROLE_SPECIFIC_CAPACITY_VARIANTS}


def test_role_specific_capacity_variants_are_configured():
    assert BASELINE_VARIANT == "all_2x32_reference"
    assert list(variants_by_id()) == [
        "all_2x32_reference",
        "advantage_3x64_policy_baseline_2x32",
        "advantage_2x128_policy_baseline_2x32",
        "all_3x64_reference",
    ]


def test_advantage_only_variants_keep_policy_and_baseline_fixed():
    variants = variants_by_id()

    advantage_3x64 = variants["advantage_3x64_policy_baseline_2x32"]
    assert advantage_3x64["policy_network_layers"] == [32, 32]
    assert advantage_3x64["baseline_network_layers"] == [32, 32]
    assert advantage_3x64["advantage_network_layers"] == [64, 64, 64]
    assert advantage_3x64["network_treatment"] == "advantage_only_capacity"

    advantage_2x128 = variants["advantage_2x128_policy_baseline_2x32"]
    assert advantage_2x128["policy_network_layers"] == [32, 32]
    assert advantage_2x128["baseline_network_layers"] == [32, 32]
    assert advantage_2x128["advantage_network_layers"] == [128, 128]
    assert advantage_2x128["network_treatment"] == "advantage_only_capacity"


def test_reference_variants_match_uniform_capacity_controls():
    variants = variants_by_id()

    all_2x32 = variants["all_2x32_reference"]
    assert all_2x32["policy_network_layers"] == [32, 32]
    assert all_2x32["advantage_network_layers"] == [32, 32]
    assert all_2x32["baseline_network_layers"] == [32, 32]
    assert all_2x32["network_treatment"] == "matched_control"

    all_3x64 = variants["all_3x64_reference"]
    assert all_3x64["policy_network_layers"] == [64, 64, 64]
    assert all_3x64["advantage_network_layers"] == [64, 64, 64]
    assert all_3x64["baseline_network_layers"] == [64, 64, 64]
    assert all_3x64["network_treatment"] == "all_network_capacity"
