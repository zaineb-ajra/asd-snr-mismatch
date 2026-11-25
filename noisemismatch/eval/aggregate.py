"""Aggregation of per-ID results (protocol, "Aggregation")."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

ID_KEYS: list[str] = ["machine_type", "machine_id"]


def drop_ids(per_id: pd.DataFrame, drop: Iterable[Mapping[str, str]]) -> pd.DataFrame:
    """Removes machine IDs (e.g. for the "fan without id_00" aggregation).

    Args:
        per_id: Table with ``machine_type`` and ``machine_id`` columns.
        drop: Items ``{"machine_type": ..., "machine_id": ...}``.

    Returns:
        The remaining rows.
    """
    keep = pd.Series(True, index=per_id.index)
    for item in drop:
        keep &= ~(
            (per_id["machine_type"] == item["machine_type"])
            & (per_id["machine_id"] == item["machine_id"])
        )
    return per_id[keep]


def aggregate(per_id: pd.DataFrame, value: str) -> tuple[pd.Series, float]:
    """Arithmetic mean over IDs per machine type, then over machine types.

    Args:
        per_id: One row per machine ID with ``machine_type`` and ``value``.
        value: Column to aggregate.

    Returns:
        ``(per machine type, overall)``.
    """
    per_type = per_id.groupby("machine_type")[value].mean()
    return per_type, float(per_type.mean())


def seed_summary(values: Iterable[float], ddof: int = 1) -> tuple[float, float]:
    """Mean and standard deviation over seeds.

    Args:
        values: One value per seed.
        ddof: Delta degrees of freedom of the standard deviation.

    Returns:
        ``(mean, std)``; std is 0.0 when all values are equal (e.g. EAT).
    """
    array = np.asarray(list(values), dtype=float)
    if np.ptp(array) == 0:  # identical values: exactly zero, not rounding noise
        return float(array[0]), 0.0
    return float(array.mean()), float(array.std(ddof=ddof))


def delta_auc(
    mismatched: Mapping[int, float], matched: Mapping[int, float], ddof: int = 1
) -> tuple[float, float]:
    """ΔAUC computed seed by seed, then summarized over seeds.

    ΔAUC_k = AUC_k(train s, test s') - AUC_k(train s', test s'), where both
    terms are aggregated values of seed k.

    Args:
        mismatched: Seed -> aggregated AUC of the mismatched condition.
        matched: Seed -> aggregated AUC of the matched condition at the test SNR.
        ddof: Delta degrees of freedom of the standard deviation.

    Returns:
        ``(mean, std)`` of ΔAUC over seeds.
    """
    if set(mismatched) != set(matched):
        raise ValueError("ΔAUC needs the same seeds in both conditions")
    return seed_summary((mismatched[k] - matched[k] for k in sorted(matched)), ddof)
