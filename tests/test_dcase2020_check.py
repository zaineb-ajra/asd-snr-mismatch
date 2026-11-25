"""Clip table of the OS-SCL diagnostic on the DCASE 2020 split."""

from noisemismatch.data.dataset import ORDER
from noisemismatch.dcase2020_check import clip_table


def test_clip_table_is_the_dcase_2020_split():
    table = clip_table(-6)
    assert len(table) == 18019
    counts = table.groupby(["split", "label"]).size().to_dict()
    # docs/data.md: 13,119 normal clips in train/, 1,600 normal and 3,300 anomalous
    assert counts == {
        ("train", "normal"): 13119,
        ("test", "normal"): 1600,
        ("test", "abnormal"): 3300,
    }
    assert table.groupby(["machine_type", "machine_id"]).ngroups == 16
    test_normals = table[(table["split"] == "test") & (table["y"] == 0)]
    assert test_normals.groupby(["machine_type", "machine_id"]).size().eq(100).all()
    assert (table["snr_db"] == -6).all()
    assert not table["strict_exclude"].any()
    assert (table["file_clip"] == table["clip"]).all()
    assert table[ORDER].equals(table.sort_values(ORDER)[ORDER].reset_index(drop=True))
