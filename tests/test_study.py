"""Study analysis: aggregation, seed-paired ΔAUC and error ratios on toy data."""

import numpy as np
import pandas as pd
import pytest

from noisemismatch.eval.study import (
    aggregated,
    deltas,
    error_ratios,
    matched_condition,
    over_seeds,
    per_id,
)


def _per_id_rows():
    """Two seeds, conditions 0dB, p6dB and multi, two test SNRs, two types."""
    rows = []
    rng = np.random.default_rng(0)
    for seed in (0, 1):
        for condition in ("0dB", "p6dB", "multi"):
            for snr in (0, 6):
                for machine_type, ids in (
                    ("fan", ("id_00", "id_02")),
                    ("pump", ("id_00",)),
                ):
                    for machine_id in ids:
                        rows.append(
                            {
                                "model": "m",
                                "train_condition": condition,
                                "seed": seed,
                                "test_snr_db": snr,
                                "machine_type": machine_type,
                                "machine_id": machine_id,
                                "auc": float(rng.uniform(0.5, 0.99)),
                                "pauc": float(rng.uniform(0.5, 0.99)),
                            }
                        )
    return pd.DataFrame(rows)


def test_aggregation_is_mean_over_ids_then_types():
    table = _per_id_rows()
    agg = aggregated(table)
    one = table[
        (table.train_condition == "0dB") & (table.seed == 0) & (table.test_snr_db == 6)
    ]
    fan = one[one.machine_type == "fan"]["auc"].mean()
    pump = one[one.machine_type == "pump"]["auc"].mean()
    row = agg[(agg.train_condition == "0dB") & (agg.seed == 0) & (agg.test_snr_db == 6)]
    assert row["auc"].item() == pytest.approx((fan + pump) / 2)
    dropped = aggregated(table, drop=[{"machine_type": "fan", "machine_id": "id_00"}])
    row = dropped[
        (dropped.train_condition == "0dB")
        & (dropped.seed == 0)
        & (dropped.test_snr_db == 6)
    ]
    fan_02 = one[(one.machine_type == "fan") & (one.machine_id == "id_02")][
        "auc"
    ].item()
    assert row["auc"].item() == pytest.approx((fan_02 + pump) / 2)


def test_deltas_pair_each_seed_with_the_matched_run_of_the_same_seed():
    agg = aggregated(_per_id_rows())
    result = deltas(agg)
    # matched cells left out: (0dB, 0) and (p6dB, 6)
    assert set(map(tuple, result[["train_condition", "test_snr_db"]].to_numpy())) == {
        ("0dB", 6),
        ("p6dB", 0),
        ("multi", 0),
        ("multi", 6),
    }
    for seed in (0, 1):
        cell = result[
            (result.train_condition == "0dB")
            & (result.test_snr_db == 6)
            & (result.seed == seed)
        ]
        a = agg[
            (agg.train_condition == "0dB") & (agg.test_snr_db == 6) & (agg.seed == seed)
        ]["auc"].item()
        m = agg[
            (agg.train_condition == "p6dB")
            & (agg.test_snr_db == 6)
            & (agg.seed == seed)
        ]["auc"].item()
        assert cell["delta"].item() == pytest.approx(a - m)


def test_error_ratio_is_a_ratio_of_aggregates():
    table = _per_id_rows()
    agg = aggregated(table)
    result = error_ratios(agg)
    cell = result[
        (result.train_condition == "p6dB")
        & (result.test_snr_db == 0)
        & (result.seed == 1)
    ]
    a = agg[(agg.train_condition == "p6dB") & (agg.test_snr_db == 0) & (agg.seed == 1)][
        "auc"
    ].item()
    m = agg[(agg.train_condition == "0dB") & (agg.test_snr_db == 0) & (agg.seed == 1)][
        "auc"
    ].item()
    assert cell["ratio"].item() == pytest.approx((1 - a) / (1 - m))


def test_over_seeds_uses_sample_std_and_zero_for_identical_seeds():
    frame = pd.DataFrame(
        {
            "cell": ["a"] * 3 + ["b"] * 3,
            "seed": [0, 1, 2] * 2,
            "v": [1.0, 2.0, 3.0, 0.7, 0.7, 0.7],
        }
    )
    out = over_seeds(frame, "v", ["cell"]).set_index("cell")
    assert out.loc["a", "mean"] == 2.0
    assert out.loc["a", "std"] == pytest.approx(1.0)
    assert out.loc["b", "std"] == 0.0
    assert out.loc["b", "per_seed"] == [0.7, 0.7, 0.7]


def test_matched_condition():
    assert matched_condition(-6) == "m6dB"
    assert matched_condition(6) == "p6dB"
    with pytest.raises(ValueError, match="no single-SNR condition"):
        matched_condition(3)


def test_per_id_parallel_equals_serial():
    rng = np.random.default_rng(1)
    rows = []
    for seed in (0, 1):
        for machine_id in ("id_00", "id_02"):
            for y in (0, 1):
                for _ in range(15):
                    rows.append(
                        {
                            "model": "m",
                            "train_condition": "0dB",
                            "seed": seed,
                            "test_snr_db": 0,
                            "machine_type": "fan",
                            "machine_id": machine_id,
                            "y": y,
                            "score": rng.normal(y, 1.0),
                            "strict_exclude": False,
                        }
                    )
    scores = pd.DataFrame(rows)
    boot = {"n_resamples": 50, "confidence": 0.95, "seed": 3}
    serial = per_id(scores, 0.1, boot)
    parallel = per_id(scores, 0.1, boot, workers=2)
    pd.testing.assert_frame_equal(serial, parallel)
    assert {"auc_ci_low", "auc_ci_high"} <= set(serial.columns)


def _toy_scores(seed=0):
    """Two models x conditions 0dB/p6dB x 2 seeds x test SNRs 0/6, two types."""
    rng = np.random.default_rng(seed)
    rows = []
    ids = (("fan", "id_00"), ("fan", "id_02"), ("pump", "id_00"))
    for machine_type, machine_id in ids:
        for label, y in (("normal", 0), ("abnormal", 1)):
            for clip in range(12):
                base = rng.normal(y, 1.0)
                for model in ("a", "b"):
                    for condition in ("0dB", "p6dB"):
                        for run_seed in (0, 1):
                            for snr in (0, 6):
                                rows.append(
                                    {
                                        "model": model,
                                        "train_condition": condition,
                                        "seed": run_seed,
                                        "test_snr_db": snr,
                                        "machine_type": machine_type,
                                        "machine_id": machine_id,
                                        "label": label,
                                        "clip": f"{clip:04d}",
                                        "y": y,
                                        "strict_exclude": False,
                                        # ties on purpose: rounded scores
                                        "score": round(base + rng.normal(0, 0.5), 1),
                                    }
                                )
    return pd.DataFrame(rows)


def test_rank_auc_equals_roc_auc_score_with_ties():
    from sklearn.metrics import roc_auc_score

    from noisemismatch.eval.study import rank_auc

    rng = np.random.default_rng(2)
    y = np.array([0] * 30 + [1] * 20)
    values = np.round(rng.normal(size=(50, 4)) + y[:, None], 1)
    expected = [roc_auc_score(y, values[:, j]) for j in range(4)]
    np.testing.assert_allclose(rank_auc(values, y), expected, rtol=0, atol=1e-12)


def test_paired_bootstrap_point_equals_protocol_aggregation():
    from noisemismatch.eval.study import COLUMN_KEYS, paired_bootstrap

    scores = _toy_scores()
    point, boot = paired_bootstrap(scores, 20, 7)
    agg = aggregated(per_id(scores, 0.1)).set_index(COLUMN_KEYS)["auc"]
    for column in point.columns:
        assert point[column].item() == pytest.approx(agg[column], abs=1e-12)
    assert boot.shape == (20, point.shape[1])


def test_paired_bootstrap_uses_the_same_draws_for_every_column():
    from noisemismatch.eval.study import paired_bootstrap

    scores = _toy_scores()
    # model b, seed 1, condition p6dB, test 6 dB gets exactly the scores of model a
    key = [
        "train_condition",
        "seed",
        "test_snr_db",
        "machine_type",
        "machine_id",
        "label",
        "clip",
    ]
    a = scores[scores.model == "a"].set_index(key)["score"]
    b = scores.model == "b"
    scores.loc[b, "score"] = a.reindex(scores.loc[b].set_index(key).index).to_numpy()
    _, boot = paired_bootstrap(scores, 30, 3)
    pd.testing.assert_series_equal(
        boot[("a", "p6dB", 1, 6)], boot[("b", "p6dB", 1, 6)], check_names=False
    )
    _, again = paired_bootstrap(scores, 30, 3, workers=2)
    pd.testing.assert_frame_equal(boot, again)


def test_delta_and_ratio_over_seeds_match_the_point_definitions():
    from noisemismatch.eval.study import (
        delta_over_seeds,
        paired_bootstrap,
        ratio_over_seeds,
    )

    scores = _toy_scores()
    point, _ = paired_bootstrap(scores, 5, 1)
    agg = aggregated(per_id(scores, 0.1))
    d = deltas(agg)
    r = error_ratios(agg)
    sel = (d.model == "a") & (d.train_condition == "0dB") & (d.test_snr_db == 6)
    assert delta_over_seeds(point, "a", "0dB", 6)[0] == pytest.approx(
        d[sel]["delta"].mean()
    )
    sel = (r.model == "b") & (r.train_condition == "p6dB") & (r.test_snr_db == 0)
    assert ratio_over_seeds(point, "b", "p6dB", 0)[0] == pytest.approx(
        r[sel]["ratio"].mean()
    )


def test_resample_ids_pauc_point_and_drop():
    from noisemismatch.eval.study import COLUMN_KEYS, resample_ids

    scores = _toy_scores()
    resamples = resample_ids(scores, 10, 4, max_fpr=0.1)
    table = per_id(scores, 0.1)
    for metric in ("auc", "pauc"):
        point, boot = resamples.aggregate(metric)
        agg = aggregated(table, metric).set_index(COLUMN_KEYS)[metric]
        for column in point.columns:
            assert point[column].item() == pytest.approx(agg[column], abs=1e-12)
        assert boot.shape == (10, point.shape[1])
    drop = [{"machine_type": "fan", "machine_id": "id_00"}]
    point, _ = resamples.aggregate("auc", drop)
    agg = aggregated(table, "auc", drop).set_index(COLUMN_KEYS)["auc"]
    for column in point.columns:
        assert point[column].item() == pytest.approx(agg[column], abs=1e-12)
    # the AUC draws do not depend on whether the pAUC is computed
    _, with_pauc = resamples.aggregate("auc")
    _, without = resample_ids(scores, 10, 4).aggregate("auc")
    pd.testing.assert_frame_equal(with_pauc, without)


def test_machine_type_values_are_means_over_the_ids_of_the_type():
    from noisemismatch.eval.study import COLUMN_KEYS, resample_ids

    scores = _toy_scores()
    point, boot = resample_ids(scores, 6, 2).machine_type("auc", "fan")
    table = per_id(scores, 0.1)
    expected = table[table.machine_type == "fan"].groupby(COLUMN_KEYS)["auc"].mean()
    for column in point.columns:
        assert point[column].item() == pytest.approx(expected[column], abs=1e-12)
    assert boot.shape == (6, point.shape[1])
