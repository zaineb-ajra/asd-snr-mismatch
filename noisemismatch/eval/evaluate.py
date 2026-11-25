"""Evaluation from saved per-clip scores (no model is rerun).

Everything reported per run is computed from ``scores.csv``: per-ID AUC and pAUC
(optionally with bootstrap confidence intervals), the strict identity variant
(rows flagged ``strict_exclude`` removed), and the aggregations of the protocol,
including the robustness aggregations (e.g. fan without ``id_00``).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from noisemismatch.eval.aggregate import aggregate, drop_ids
from noisemismatch.eval.metrics import auc, bootstrap_auc, pauc

GROUP: list[str] = ["test_snr_db", "machine_type", "machine_id"]


def per_id_table(
    scores: pd.DataFrame,
    max_fpr: float,
    strict: bool = False,
    bootstrap: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """AUC and pAUC per test SNR and machine ID.

    Args:
        scores: Rows of ``scores.csv`` (``y``, ``score``, ``strict_exclude`` and
            the grouping columns).
        max_fpr: pAUC bound.
        strict: Remove the rows of the strict identity variant.
        bootstrap: ``evaluation.bootstrap`` of the protocol, to add
            ``auc_ci_low`` / ``auc_ci_high``; None to skip.

    Returns:
        One row per (test SNR, machine type, machine ID).
    """
    if strict:
        scores = scores[~scores["strict_exclude"].astype(bool)]
    rows = []
    for (snr, machine_type, machine_id), part in scores.groupby(GROUP, sort=True):
        y, s = part["y"].to_numpy(), part["score"].to_numpy()
        row = {
            "test_snr_db": int(snr),
            "machine_type": machine_type,
            "machine_id": machine_id,
            "n_test_normal": int((y == 0).sum()),
            "n_test_anomalous": int((y == 1).sum()),
            "auc": auc(y, s),
            "pauc": pauc(y, s, max_fpr),
        }
        if bootstrap is not None:
            row["auc_ci_low"], row["auc_ci_high"] = bootstrap_auc(
                y,
                s,
                int(bootstrap["n_resamples"]),
                float(bootstrap["confidence"]),
                int(bootstrap["seed"]),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregated(
    per_id: pd.DataFrame, robustness: list[dict[str, Any]] | None = None
) -> pd.DataFrame:
    """Overall AUC and pAUC per test SNR, main and robustness aggregations.

    Args:
        per_id: Output of :func:`per_id_table`.
        robustness: ``aggregation.robustness`` of the protocol.

    Returns:
        One row per (aggregation, test SNR) with ``auc`` and ``pauc``.
    """
    variants = [("main", [])] + [(r["name"], r["drop"]) for r in robustness or []]
    rows = []
    for name, drop in variants:
        kept = drop_ids(per_id, drop)
        for snr, part in kept.groupby("test_snr_db", sort=True):
            rows.append(
                {
                    "aggregation": name,
                    "test_snr_db": int(snr),
                    "auc": aggregate(part, "auc")[1],
                    "pauc": aggregate(part, "pauc")[1],
                }
            )
    return pd.DataFrame(rows)
