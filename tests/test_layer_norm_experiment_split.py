from experiments.leduc_poker.dream_layer_norm_network_ablation.config import (
    EXPERIMENT_CONFIG as LAYER_NORM_CONFIG,
    LAYER_NORM_EXPERIMENT_VARIANTS,
)
from experiments.leduc_poker.dream_plain_network_depth_ablation.config import (
    EXPERIMENT_CONFIG as PLAIN_DEPTH_CONFIG,
    PLAIN_NETWORK_DEPTH_VARIANTS,
)
from experiments.leduc_poker.dream_residual_layer_norm_network_ablation.config import (
    EXPERIMENT_CONFIG as RESIDUAL_LAYER_NORM_CONFIG,
    RESIDUAL_LAYER_NORM_NETWORK_VARIANTS,
)


def variant_ids(variants):
    return [variant["variant_id"] for variant in variants]


def test_experiment_17_plain_depth_variants_are_isolated():
    assert variant_ids(PLAIN_NETWORK_DEPTH_VARIANTS) == [
        "plain_layers2_width32",
        "plain_layers4_width32",
        "plain_layers8_width32",
    ]


def test_experiment_18_layer_norm_variants_are_isolated():
    assert variant_ids(LAYER_NORM_EXPERIMENT_VARIANTS) == [
        "plain_layers2_width32",
        "layer_norm_layers2_width32",
        "layer_norm_layers4_width32",
        "layer_norm_layers8_width32",
    ]


def test_experiment_19_residual_layer_norm_variants_are_isolated():
    assert variant_ids(RESIDUAL_LAYER_NORM_NETWORK_VARIANTS) == [
        "plain_layers2_width32",
        "residual_layer_norm_layers2_width32",
        "residual_layer_norm_layers4_width32",
        "residual_layer_norm_layers8_width32",
    ]


def test_split_experiments_have_distinct_names_and_output_roots():
    configs = [PLAIN_DEPTH_CONFIG, LAYER_NORM_CONFIG, RESIDUAL_LAYER_NORM_CONFIG]

    assert [config["experiment_name"] for config in configs] == [
        "leduc_poker_dream_plain_network_depth_ablation",
        "leduc_poker_dream_layer_norm_network_ablation",
        "leduc_poker_dream_residual_layer_norm_network_ablation",
    ]
    assert len({config["plot_prefix"] for config in configs}) == 3
    assert len({config["output_root"] for config in configs}) == 3
