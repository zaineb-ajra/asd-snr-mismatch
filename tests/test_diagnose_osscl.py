import numpy as np
import pytest

from noisemismatch.diagnose_osscl import geometry


def test_geometry_separates_a_clean_group_from_a_missing_one():
    # class a: tight group on the x axis; class b: on the y axis, but its test
    # clips sit on the x axis (a clean group in the wrong place)
    z_train = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    y_train = np.array([0, 0, 1, 1])
    z_test = np.array([[1.0, 0.0], [1.0, 0.0]])
    y_test = np.array([0, 1])
    centers = np.array([[1.0, 0.0], [0.0, 1.0]])
    a, b = geometry(["a", "b"], z_train, y_train, z_test, y_test, centers)
    assert a["train_centroid_accuracy"] == 1.0
    assert a["test_centroid_accuracy"] == 1.0
    assert b["train_centroid_accuracy"] == 1.0
    assert b["test_centroid_accuracy"] == 0.0  # predicted as "a"
    assert b["test_cosine_to_own_centroid"] == pytest.approx(0.0)
    assert a["nearest_other_centroid"] == "b"
    assert a["cosine_centroid_to_arcface_center"] == pytest.approx(1.0)
