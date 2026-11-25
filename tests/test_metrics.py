import numpy as np
import pandas as pd
import pytest

from noisemismatch.eval.aggregate import aggregate, delta_auc, drop_ids, seed_summary
from noisemismatch.eval.metrics import auc, bootstrap_auc, pauc

NEGATIVES = np.round(np.arange(10) * 0.1, 1)  # 0.0, 0.1, ..., 0.9


def test_auc_counts_correctly_ordered_pairs():
    # Pairs (anomalous, normal): (0.35, 0.1) ok, (0.35, 0.4) not, (0.8, *) ok.
    assert auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(3 / 4)


def test_auc_ties_count_half():
    assert auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)


def test_pauc_perfect_detector_is_one():
    assert pauc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_pauc_mcclish_standardization_by_hand():
    # One anomaly ranked below the highest of 10 normal clips: the ROC curve has
    # TPR = 0 on FPR in [0, 0.1], raw partial area 0. McClish:
    # 0.5 * (1 + (0 - 0.1**2 / 2) / (0.1 - 0.1**2 / 2)) = 0.5 * (1 - 0.005 / 0.095)
    labels = np.r_[np.zeros(10), 1]
    scores = np.r_[NEGATIVES, 0.85]
    assert pauc(labels, scores) == pytest.approx(0.5 * (1 - 0.005 / 0.095))


def test_pauc_partial_area_by_hand():
    # Anomalies at 0.95 (above all) and 0.85: TPR = 0.5 on FPR in [0, 0.1],
    # raw area 0.05 -> 0.5 * (1 + (0.05 - 0.005) / 0.095).
    labels = np.r_[np.zeros(10), 1, 1]
    scores = np.r_[NEGATIVES, 0.95, 0.85]
    assert pauc(labels, scores) == pytest.approx(0.5 * (1 + 0.045 / 0.095))


def test_bootstrap_is_deterministic_and_covers_the_estimate():
    rng = np.random.default_rng(0)
    labels = np.r_[np.zeros(100), np.ones(100)]
    scores = np.r_[rng.normal(0, 1, 100), rng.normal(1, 1, 100)]
    ci = bootstrap_auc(labels, scores, 200, 0.95, seed=3)
    assert ci == bootstrap_auc(labels, scores, 200, 0.95, seed=3)
    assert ci[0] < auc(labels, scores) < ci[1]
    perfect = bootstrap_auc([0, 0, 1, 1], [0, 0.1, 0.8, 0.9], 50, 0.95, seed=0)
    assert perfect == (1.0, 1.0)


def _per_id():
    return pd.DataFrame(
        {
            "machine_type": ["fan", "fan", "pump", "valve", "valve"],
            "machine_id": ["id_00", "id_02", "id_00", "id_02", "id_04"],
            "auc": [0.6, 0.8, 0.9, 0.5, 0.7],
        }
    )


def test_aggregate_means_ids_then_types():
    per_type, overall = aggregate(_per_id(), "auc")
    assert per_type.to_dict() == pytest.approx({"fan": 0.7, "pump": 0.9, "valve": 0.6})
    assert overall == pytest.approx((0.7 + 0.9 + 0.6) / 3)


def test_robustness_aggregation_without_an_id():
    kept = drop_ids(_per_id(), [{"machine_type": "fan", "machine_id": "id_00"}])
    per_type, overall = aggregate(kept, "auc")
    assert per_type["fan"] == pytest.approx(0.8)
    assert overall == pytest.approx((0.8 + 0.9 + 0.6) / 3)


def test_delta_auc_pairs_seeds():
    mean, std = delta_auc({0: 0.70, 1: 0.60, 2: 0.65}, {0: 0.80, 1: 0.80, 2: 0.70})
    # Per seed: -0.10, -0.20, -0.05.
    assert mean == pytest.approx(-0.35 / 3)
    assert std == pytest.approx(np.std([-0.10, -0.20, -0.05], ddof=1))
    with pytest.raises(ValueError, match="same seeds"):
        delta_auc({0: 0.7}, {1: 0.8})


def test_seed_summary_identical_values():
    assert seed_summary([0.8, 0.8, 0.8]) == (pytest.approx(0.8), 0.0)
