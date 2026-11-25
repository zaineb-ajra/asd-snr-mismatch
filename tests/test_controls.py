from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from noisemismatch.controls import id_confusion, score_summary
from noisemismatch.data.dataset import ClipSet


class _Model:
    classes: ClassVar[list[str]] = ["id_00", "id_02", "id_04"]

    def __init__(self, predicted):
        self.predicted = predicted

    def id_probabilities(self, clips):
        probabilities = np.full((len(clips), 2, len(self.classes)), 0.1)
        for k, name in enumerate(self.predicted):
            probabilities[k, :, self.classes.index(name)] = 0.8
        return probabilities


def _clips(machine_ids, labels=None):
    table = pd.DataFrame({"machine_type": "fan", "machine_id": machine_ids})
    table["label"] = labels if labels is not None else "normal"
    table["y"] = (table["label"] == "abnormal").astype(int)
    return ClipSet(table, audio=None)


def test_id_confusion_counts_every_class():
    clips = _clips(["id_00", "id_00", "id_02"])
    matrix = id_confusion(_Model(["id_00", "id_02", "id_02"]), clips)
    assert list(matrix.index) == ["id_00", "id_02", "id_04"]
    assert list(matrix.columns) == ["id_00", "id_02", "id_04"]
    assert matrix.loc["id_00"].tolist() == [1, 1, 0]
    assert matrix.loc["id_02"].tolist() == [0, 1, 0]
    assert matrix.loc["id_04"].tolist() == [0, 0, 0]  # class without test clips


def test_score_summary_per_id():
    clips = _clips(
        ["id_00"] * 4 + ["id_02"] * 2,
        ["normal", "normal", "abnormal", "abnormal", "normal", "abnormal"],
    )
    summary = score_summary(clips, np.array([0.1, 0.2, 0.9, 0.8, 0.5, 0.4]))
    assert [s["machine_id"] for s in summary] == ["id_00", "id_02"]
    assert summary[0]["auc"] == pytest.approx(1.0)
    assert summary[1]["auc"] == pytest.approx(0.0)
    assert summary[0]["normal_quartiles"][1] == pytest.approx(0.15)
