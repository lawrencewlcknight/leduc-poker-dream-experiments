
"""Shared experiment utilities."""

from pathlib import Path
from typing import Mapping, Optional, Sequence, Union
import numpy as np

from dream_poker.constants import KUHN_AVERAGE_POLICY_VALUE_TARGET


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_mean(xs: Sequence[float]) -> float:
    arr = np.asarray(xs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else float("nan")


def safe_std(xs: Sequence[float]) -> float:
    arr = np.asarray(xs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")


def standard_error(xs: Sequence[float]) -> float:
    arr = np.asarray(xs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return 0.0 if len(arr) == 1 else float("nan")
    return float(arr.std(ddof=1) / np.sqrt(len(arr)))


def compute_auc(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 2:
        return float("nan")
    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]
    denom = float(x_arr[-1] - x_arr[0])
    if denom <= 0:
        return float("nan")
    return float(np.trapz(y_arr, x_arr) / denom)


def average_policy_value_from_player_0(player_0_value: float) -> float:
    return float(player_0_value)


def average_policy_value_target(config: Optional[Mapping] = None) -> float:
    if config is not None and "average_policy_value_target" in config:
        return float(config["average_policy_value_target"])
    return float(KUHN_AVERAGE_POLICY_VALUE_TARGET)


def ensure_average_policy_value_columns(
    df,
    target: Optional[float] = None,
    *,
    inplace: bool = False,
):
    result = df if inplace else df.copy()
    if "average_policy_value" not in result.columns and "policy_value_player_0" in result.columns:
        result["average_policy_value"] = result["policy_value_player_0"].astype(float)
    if "average_policy_value_error" not in result.columns and "average_policy_value" in result.columns:
        value_target = float(target) if target is not None else average_policy_value_target()
        result["average_policy_value_error"] = (
            result["average_policy_value"].astype(float) - value_target
        ).abs()
    return result


def grad_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            val = float(p.grad.detach().data.norm(2).cpu().item())
            total += val * val
    return float(np.sqrt(total))
