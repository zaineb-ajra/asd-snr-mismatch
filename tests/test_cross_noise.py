import numpy as np
import pandas as pd
import pytest

from noisemismatch.data.cross_noise import (
    largest_difference,
    partner_statistics,
    summarize,
)


def test_partner_statistics_separates_other_and_all_types():
    scores = np.array([[0.9, 0.6, -0.2], [0.1, 0.2, -0.7]])
    stats = partner_statistics(
        scores, np.array(["fan", "pump"]), np.array(["fan", "pump", "valve"])
    )
    # Test file 0 (fan): other types = pump, valve.
    assert stats.loc[0, "max_other_types"] == pytest.approx(0.6)
    assert stats.loc[0, "max_all_types"] == pytest.approx(0.9)
    # Test file 1 (pump): other types = fan, valve.
    assert stats.loc[1, "max_other_types"] == pytest.approx(0.1)
    assert stats.loc[1, "min_other_types"] == pytest.approx(-0.7)


def test_summarize_and_largest_difference():
    details = pd.DataFrame(
        {
            "split_name": ["main"] * 4,
            "snr_db": [0] * 4,
            "label": ["normal", "normal", "abnormal", "abnormal"],
            "max_other_types": [0.6, 0.1, 0.1, 0.1],
            "min_other_types": [-0.1, -0.6, -0.1, -0.1],
            "max_all_types": [0.6, 0.7, 0.1, 0.1],
            "min_all_types": [-0.1, -0.6, -0.1, -0.1],
        }
    )
    summary = summarize(details, threshold=0.5).set_index(["scope", "label"])
    assert summary.loc[("other_types", "normal"), "share"] == 0.5
    assert summary.loc[("other_types", "normal"), "n_negative"] == 1
    assert summary.loc[("all_types", "normal"), "share"] == 1.0
    assert summary.loc[("all_types", "abnormal"), "share"] == 0.0
    assert largest_difference(summary.reset_index()) == pytest.approx(1.0)
