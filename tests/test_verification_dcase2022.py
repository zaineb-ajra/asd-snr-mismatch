import io
import zipfile

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from noisemismatch.verification_dcase2022 import (
    DcaseAudio,
    clip_table,
    compare,
    section_metrics,
)


def test_clip_table_parses_names_and_skips_attribute_files(tmp_path):
    path = tmp_path / "dev_fan.zip"
    names = [
        "fan/train/section_00_source_train_normal_0000_m-n_W.wav",
        "fan/train/section_00_target_train_normal_0000_m-n_X.wav",
        "fan/test/section_01_target_test_anomaly_0003_m-n_X.wav",
    ]
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("fan/attributes_00.csv", "x")
        for name in names:
            buffer = io.BytesIO()
            sf.write(
                buffer,
                np.zeros(16000, dtype=np.int16),
                16000,
                format="WAV",
                subtype="PCM_16",
            )
            zf.writestr(name, buffer.getvalue())
    table = clip_table(path)
    assert len(table) == 3
    row = table[table["split"] == "test"].iloc[0]
    assert (row["machine_id"], row["label"], row["domain"], row["y"]) == (
        "section_01",
        "abnormal",
        "target",
        1,
    )
    audio = DcaseAudio(path, table)
    assert audio.load("fan", 0, table.iloc[[2, 0]]).shape == (2, 16000)


def test_section_metrics_use_all_anomalies_per_domain():
    scores = pd.DataFrame(
        {
            "machine_id": ["section_00"] * 6,
            "domain": ["source", "source", "target", "target", "source", "target"],
            "y": [0, 0, 0, 0, 1, 1],
            "score": [0.1, 0.2, 0.9, 0.3, 0.5, 0.6],
        }
    )
    row = section_metrics(scores, 0.1).iloc[0]
    # Source normals (0.1, 0.2) vs both anomalies (0.5, 0.6): perfect.
    assert row["auc_source"] == pytest.approx(100.0)
    # Target normals (0.9, 0.3) vs anomalies (0.5, 0.6): 2 of 4 pairs ordered.
    assert row["auc_target"] == pytest.approx(50.0)


def test_compare_applies_the_preregistered_rules():
    summary = pd.DataFrame(
        {
            "machine_type": ["valve", "valve", "valve"],
            "section": ["02", "02", "02"],
            "metric": ["auc_source", "auc_target", "pauc"],
            "mean": [78.0, 70.0, 85.7],
            "std": [1.0, 2.0, 0.5],
        }
    )
    table = compare(summary)
    assert len(table) == 10  # all retained cells are listed
    valve = table[table["machine_type"] == "valve"].set_index("metric")
    assert bool(valve.loc["auc_source", "agrees"])  # 78.0 vs 76.26
    assert not bool(valve.loc["auc_target", "agrees"])  # 70.0 vs 78.74
    assert bool(valve.loc["pauc", "agrees"])
    assert table[table["machine_type"] != "valve"]["our_mean"].isna().all()
