import pandas as pd
import pytest

from noisemismatch.data.split import (
    ANOMALOUS,
    NORMAL,
    TEST,
    TRAIN,
    check_split,
    make_cluster_split,
    split_balance,
)


def _table(rows):
    return pd.DataFrame(
        rows, columns=["machine_type", "machine_id", "label", "clip", "cluster"]
    )


def _singletons(n_normal, n_anomalous, machine_id="id_00"):
    rows = [
        ("fan", machine_id, NORMAL, f"{i:08d}", f"{machine_id}-n{i}")
        for i in range(n_normal)
    ]
    rows += [
        ("fan", machine_id, ANOMALOUS, f"{i:08d}", f"{machine_id}-a{i}")
        for i in range(n_anomalous)
    ]
    return rows


def test_singletons_give_the_mimii_protocol():
    split = make_cluster_split(_table(_singletons(20, 5)), seed=0)
    check_split(split)  # exact balance required
    balance = split_balance(split)
    assert balance[
        ["train_normal", "test_normal", "test_abnormal"]
    ].values.tolist() == [[15, 5, 5]]


def test_clusters_are_never_split_and_follow_their_anomalies():
    rows = _singletons(20, 5)
    # Normal clips 0-2 form one cluster; normal 3 shares a cluster with anomaly 0.
    rows = [(*r[:4], "big") if r[2] == NORMAL and int(r[3]) < 3 else r for r in rows]
    rows = [
        (*r[:4], "mixed") if r[3] == "00000000" and r[2] == ANOMALOUS else r
        for r in rows
    ]
    rows = [
        (*r[:4], "mixed") if r[3] == "00000003" and r[2] == NORMAL else r for r in rows
    ]
    split = make_cluster_split(_table(rows), seed=0)
    check_split(split, require_balance=False)
    assert split.groupby("cluster")["split"].nunique().max() == 1
    assert (split.loc[split["cluster"] == "mixed", "split"] == TEST).all()
    assert (split_balance(split)["test_difference"] == 0).all()


def test_closest_count_when_exact_is_impossible():
    # 3 anomalies; normal clips only in clusters of 2 -> 2 or 4 normal test clips.
    rows = [("fan", "id_00", ANOMALOUS, f"{i:08d}", f"a{i}") for i in range(3)]
    rows += [("fan", "id_00", NORMAL, f"{i:08d}", f"pair{i // 2}") for i in range(10)]
    split = make_cluster_split(_table(rows), seed=0)
    check_split(split, require_balance=False)
    difference = split_balance(split)["test_difference"].iloc[0]
    assert abs(difference) == 1
    with pytest.raises(ValueError, match="normal vs"):
        check_split(split)


def test_cluster_split_is_deterministic_and_order_independent():
    table = _table(_singletons(30, 6) + _singletons(25, 4, machine_id="id_02"))
    shuffled = table.sample(frac=1.0, random_state=1).reset_index(drop=True)
    assert make_cluster_split(table, 3).equals(make_cluster_split(shuffled, 3))
    assert not make_cluster_split(table, 3).equals(make_cluster_split(table, 4))


def test_cluster_across_machine_ids_is_rejected():
    rows = _singletons(5, 1) + _singletons(5, 1, machine_id="id_02")
    rows[0] = (*rows[0][:4], "shared")
    rows[6] = (*rows[6][:4], "shared")
    with pytest.raises(ValueError, match="spans several machine IDs"):
        make_cluster_split(_table(rows), 0)


def test_split_cluster_is_detected():
    split = make_cluster_split(_table(_singletons(20, 5)), seed=0)
    normal = split.index[split["label"] == NORMAL]
    split.loc[normal[:2], "cluster"] = "x"
    split.loc[normal[0], "split"] = TRAIN
    split.loc[normal[1], "split"] = TEST
    with pytest.raises(ValueError, match="split across sets"):
        check_split(split, require_balance=False)


def test_max_share_difference_keeps_splits_apart():
    from noisemismatch.data.build_split import max_share_difference

    check = pd.DataFrame(
        {
            "split_name": ["a", "a", "b", "b"],
            "snr_db": [6, 6, 6, 6],
            "scope": ["machine_id"] * 4,
            "label": ["normal", "abnormal"] * 2,
            "share": [0.03, 0.0, 0.01, 0.0],
        }
    )
    assert max_share_difference(check) == pytest.approx(0.03)
