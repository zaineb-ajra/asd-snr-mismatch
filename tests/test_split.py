import pandas as pd
import pytest

from noisemismatch.data.split import (
    ANOMALOUS,
    IDENTITY_COLUMNS,
    NORMAL,
    TEST,
    TRAIN,
    assign_files,
    check_no_leakage,
    check_split,
    load_split,
    make_split,
    save_split,
)

SNRS = (-6, 0, 6)


def _identities(
    n_normal=20, n_anomalous=5, groups=(("fan", "id_00"), ("pump", "id_02"))
):
    rows = []
    for machine_type, machine_id in groups:
        rows += [
            (machine_type, machine_id, NORMAL, f"{i:08d}") for i in range(n_normal)
        ]
        rows += [
            (machine_type, machine_id, ANOMALOUS, f"{i:08d}")
            for i in range(n_anomalous)
        ]
    return pd.DataFrame(rows, columns=list(IDENTITY_COLUMNS))


def _files(identities, snrs=SNRS):
    return pd.concat([identities.assign(snr_db=snr) for snr in snrs], ignore_index=True)


def test_split_follows_the_mimii_protocol():
    split = make_split(_identities(), seed=1)
    check_split(split)
    for _, group in split.groupby(["machine_type", "machine_id"]):
        test = group[group["split"] == TEST]
        assert (test["label"] == ANOMALOUS).sum() == 5
        assert (test["label"] == NORMAL).sum() == 5
        assert (group["split"] == TRAIN).sum() == 15
        assert (group.loc[group["split"] == TRAIN, "label"] == NORMAL).all()


def test_split_is_deterministic_and_seed_dependent():
    ids = _identities()
    assert make_split(ids, 1).equals(make_split(ids, 1))
    assert not make_split(ids, 1).equals(make_split(ids, 2))


def test_split_ignores_input_row_order():
    ids = _identities()
    shuffled = ids.sample(frac=1.0, random_state=0).reset_index(drop=True)
    assert make_split(ids, 1).equals(make_split(shuffled, 1))


def test_a_group_draw_does_not_depend_on_other_groups():
    both = make_split(_identities(), 1)
    fan_only = make_split(_identities(groups=(("fan", "id_00"),)), 1)
    fan_from_both = both[both["machine_type"] == "fan"].reset_index(drop=True)
    assert fan_from_both.equals(fan_only)


def test_split_rejects_duplicates_and_bad_groups():
    ids = _identities()
    with pytest.raises(ValueError, match="duplicated"):
        make_split(pd.concat([ids, ids.iloc[:1]]), 1)
    with pytest.raises(ValueError, match="anomalous clips but only"):
        make_split(_identities(n_normal=3, n_anomalous=5), 1)


def test_check_split_detects_violations():
    split = make_split(_identities(), 1)
    bad = split.copy()
    bad.loc[bad["label"] == ANOMALOUS, "split"] = TRAIN
    with pytest.raises(ValueError, match="anomalous clip in the training set"):
        check_split(bad)
    unbalanced = split.copy()
    first_test_normal = unbalanced.index[
        (unbalanced["split"] == TEST) & (unbalanced["label"] == NORMAL)
    ][0]
    unbalanced.loc[first_test_normal, "split"] = TRAIN
    with pytest.raises(ValueError, match="normal vs"):
        check_split(unbalanced)


def test_files_inherit_the_split_of_their_clip_at_every_snr():
    ids = _identities()
    split = make_split(ids, 1)
    files = assign_files(split, _files(ids))
    check_no_leakage(files)
    per_clip = files.groupby(list(IDENTITY_COLUMNS))["split"].nunique()
    assert (per_clip == 1).all()
    assert len(files) == len(SNRS) * len(ids)


def test_assign_files_requires_every_clip_at_every_snr():
    ids = _identities()
    split = make_split(ids, 1)
    files = _files(ids)
    files = files.drop(index=files.index[(files["snr_db"] == 6)][:1])
    with pytest.raises(ValueError, match="not present at every SNR"):
        assign_files(split, files)


def test_assign_files_rejects_unknown_files():
    ids = _identities()
    split = make_split(ids, 1)
    extra = pd.DataFrame(
        [("fan", "id_00", NORMAL, "99999999", 0)],
        columns=[*IDENTITY_COLUMNS, "snr_db"],
    )
    with pytest.raises(ValueError, match="no entry in the split"):
        assign_files(split, pd.concat([_files(ids), extra], ignore_index=True))


def test_leak_across_snrs_is_detected():
    ids = _identities()
    files = assign_files(make_split(ids, 1), _files(ids))
    # A file-level split would allow this: a test clip used for training at +6 dB.
    leak = files.index[(files["split"] == TEST) & (files["snr_db"] == 6)][0]
    files.loc[leak, "split"] = TRAIN
    with pytest.raises(ValueError, match="both training and test"):
        check_no_leakage(files)


def test_save_and_load_roundtrip(tmp_path):
    split = make_split(_identities(), 1)
    path = tmp_path / "split.csv"
    save_split(split, path)
    loaded = load_split(path)
    assert loaded.equals(split.astype(str))
    assert "\r" not in path.read_text()
