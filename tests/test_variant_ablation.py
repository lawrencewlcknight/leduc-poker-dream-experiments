import pandas as pd

from dream_poker.variant_ablation import add_variant_curve_columns


def test_add_variant_curve_columns_broadcasts_list_metadata():
    curves = pd.DataFrame({"iteration": range(7), "exploitability": [1.0] * 7})
    config = {
        "execution_backend": "sequential",
        "policy_network_layers": [64, 64, 64],
    }
    variant = {"variant_id": "dream_3x64_sequential", "label": "DREAM 3x64 sequential"}

    result = add_variant_curve_columns(
        curves,
        config,
        variant,
        ["execution_backend", "policy_network_layers"],
    )

    assert result["execution_backend"].tolist() == ["sequential"] * 7
    assert result["policy_network_layers"].tolist() == [[64, 64, 64]] * 7
