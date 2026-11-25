"""Mechanical application of the pre-registered selection of validation cells."""

import pandas as pd

from noisemismatch.config import REPO_ROOT

TABLE = REPO_ROOT / "metadata" / "published" / "dohi2022_table2_mobilenetv2.csv"
MAX_STD = 3.0


def retained(table: pd.DataFrame, metric: str) -> list[str]:
    rows = table[table[f"{metric}_std"] <= MAX_STD]
    return [
        f"{t} {s}" for t, s in zip(rows["machine_type"], rows["section"], strict=True)
    ]


def test_selected_cells_are_the_documented_ones():
    table = pd.read_csv(TABLE, dtype={"section": str})
    assert len(table) == 21
    assert retained(table, "auc_source") == ["slider 00", "valve 02"]
    assert retained(table, "auc_target") == ["valve 02"]
    assert retained(table, "pauc") == [
        "ToyCar 00",
        "ToyCar 01",
        "ToyTrain 00",
        "fan 02",
        "slider 01",
        "slider 02",
        "valve 02",
    ]
