import numpy as np
import pandas as pd
import pytest

from noisemismatch.verification_wilkinghoff import (
    harmonic_means,
    section_metrics,
    with_classes,
)


def test_class_labels_from_file_names():
    table = pd.DataFrame(
        {
            "machine_type": ["gearbox", "fan"],
            "machine_id": ["section_00", "section_02"],
            "clip": [
                "section_00_source_train_normal_0000_volt_3.0",
                "section_02_target_train_normal_0003_n-lv_L3",
            ],
        }
    )
    labels = with_classes(table)["class_label"].tolist()
    assert labels == ["gearbox/section_00/volt_3.0", "fan/section_02/n-lv_L3"]


def test_section_metrics_both_definitions():
    # source: normals 0.1, 0.2; anomaly 0.9. target: normal 0.95; anomaly 0.3
    scores = pd.DataFrame(
        {
            "machine_type": "fan",
            "machine_id": "section_00",
            "domain": ["source", "source", "source", "target", "target"],
            "y": [0, 0, 1, 0, 1],
            "score": [0.1, 0.2, 0.9, 0.95, 0.3],
        }
    )
    row = section_metrics(scores, 0.1).iloc[0]
    assert row["auc_source_restricted"] == pytest.approx(100.0)
    # official source: normals 0.1, 0.2 vs anomalies 0.9, 0.3 -> 4 of 4 pairs
    assert row["auc_source_official"] == pytest.approx(100.0)
    assert row["auc_target_restricted"] == pytest.approx(0.0)
    # official target: normal 0.95 vs anomalies 0.9, 0.3 -> 0 of 2 pairs
    assert row["auc_target_official"] == pytest.approx(0.0)
    # all: normals 0.1, 0.2, 0.95 vs anomalies 0.9, 0.3 -> 4 of 6 pairs
    assert row["auc_mixed"] == pytest.approx(100 * 4 / 6)


def test_harmonic_means_over_sections():
    sections = pd.DataFrame({"auc_mixed": [50.0, 100.0], "pauc_mixed": [60.0, 60.0]})
    means = harmonic_means(sections)
    assert means["auc_mixed"] == pytest.approx(2 / (1 / 50 + 1 / 100))
    assert means["pauc_mixed"] == pytest.approx(60.0)
    assert np.isfinite(list(means.values())).all()


def test_member_names_with_and_without_labels():
    from noisemismatch.verification_wilkinghoff import NAME

    m = NAME.fullmatch("fan/train/section_02_target_train_normal_0003_n-lv_L3.wav")
    assert m["domain"] == "target"
    assert m["label"] == "normal"
    assert m["attributes"] == "n-lv_L3"
    m = NAME.fullmatch("gearbox/test/section_00_source_test_anomaly_0000_volt_3.0.wav")
    assert m["attributes"] == "volt_3.0"
    m = NAME.fullmatch("bandsaw/train/section_00_source_train_normal_0000.wav")
    assert m["attributes"] is None
    m = NAME.fullmatch("valve/test/section_04_0012.wav")  # DCASE 2022 evaluation test
    assert m["section"] == "04"
    assert m["label"] is None
    assert m["clip"] == "section_04_0012"


def test_compare_marks_criterion_and_information_cells():
    from noisemismatch.verification_wilkinghoff import compare

    rows = []
    for part in ("dev", "eval"):
        for metric in (
            "auc_source_official",
            "auc_target_official",
            "auc_mixed",
            "pauc_mixed",
            "auc_source_restricted",
            "auc_target_restricted",
            "pauc_source_official",
            "pauc_source_restricted",
            "pauc_target_official",
            "pauc_target_restricted",
        ):
            rows.append(
                {"set": part, "metric": metric, "mean": 80.0, "std": 1.0, "n": 5}
            )
    table = compare(pd.DataFrame(rows), "2022")
    assert int(table["criterion"].sum()) == 8  # 4 cells x (dev, eval)
    assert len(table) == 8 + 12
    dev_auc = table[
        (table["set"] == "dcase2022_dev") & (table["our_metric"] == "auc_mixed")
    ]
    assert float(dev_auc["difference"].iloc[0]) == pytest.approx(80.0 - 82.5)
    assert bool(dev_auc["agrees"].iloc[0])


def test_test_only_classes_come_from_named_test_clips():
    from noisemismatch.verification_wilkinghoff import test_only_classes

    table = pd.DataFrame(
        {
            "split": ["train", "train", "test", "test", "test"],
            "attributes": ["vel_1", "vel_2", "vel_2", "vel_9", None],
            "class_label": [
                "s/00/vel_1",
                "s/00/vel_2",
                "s/00/vel_2",
                "s/00/vel_9",
                "s/03/",
            ],
        }
    )
    assert test_only_classes(table) == ["s/00/vel_9"]  # unnamed test clips ignored
