"""Strict variant, bootstrap and robustness aggregation from saved scores only."""

import numpy as np
import pandas as pd
import pytest

from noisemismatch.eval.evaluate import aggregated, per_id_table

BOOTSTRAP = {"n_resamples": 100, "confidence": 0.95, "seed": 0}


def _scores():
    rng = np.random.default_rng(0)
    rows = []
    for machine_type, machine_id in (
        ("fan", "id_00"),
        ("fan", "id_02"),
        ("pump", "id_00"),
    ):
        for y in (0, 1):
            for k in range(20):
                rows.append(
                    {
                        "test_snr_db": 6,
                        "machine_type": machine_type,
                        "machine_id": machine_id,
                        "y": y,
                        "score": rng.normal(y, 1.0),
                        "strict_exclude": k < 5,
                    }
                )
    return pd.DataFrame(rows)


def test_strict_variant_removes_flagged_rows():
    full = per_id_table(_scores(), 0.1)
    strict = per_id_table(_scores(), 0.1, strict=True)
    assert (full["n_test_normal"] == 20).all()
    assert (strict["n_test_normal"] == 15).all()
    assert (strict["n_test_anomalous"] == 15).all()


def test_bootstrap_columns():
    table = per_id_table(_scores(), 0.1, bootstrap=BOOTSTRAP)
    assert (table["auc_ci_low"] <= table["auc"]).all()
    assert (table["auc"] <= table["auc_ci_high"]).all()


def test_robustness_aggregation_from_the_same_per_id_values():
    per_id = per_id_table(_scores(), 0.1)
    out = aggregated(
        per_id,
        [
            {
                "name": "fan_without_id_00",
                "drop": [{"machine_type": "fan", "machine_id": "id_00"}],
            }
        ],
    ).set_index("aggregation")
    auc = per_id.set_index(["machine_type", "machine_id"])["auc"]
    main = (
        (auc[("fan", "id_00")] + auc[("fan", "id_02")]) / 2 + auc[("pump", "id_00")]
    ) / 2
    robust = (auc[("fan", "id_02")] + auc[("pump", "id_00")]) / 2
    assert out.loc["main", "auc"] == pytest.approx(main)
    assert out.loc["fan_without_id_00", "auc"] == pytest.approx(robust)
