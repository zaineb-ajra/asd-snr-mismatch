"""Leakage control with content-based identities (synthetic data)."""

import pandas as pd
import pytest

from noisemismatch.data.split import (
    IDENTITY_COLUMNS,
    TEST,
    TRAIN,
    assign_files,
    check_content_identity,
    check_no_leakage,
    check_split,
    identity_files,
)


def _identities(m6_clips):
    """Four normal and two abnormal identities of fan/id_00; -6 dB files given."""
    refs = [("normal", f"{i:08d}") for i in range(4)] + [
        ("abnormal", f"{i:08d}") for i in range(2)
    ]
    rows = []
    for (label, clip), (m6_label, m6_clip) in zip(refs, m6_clips, strict=True):
        rows.append(
            {
                "machine_type": "fan",
                "machine_id": "id_00",
                "label": label,
                "clip": clip,
                "p6_machine_id": "id_00",
                "p6_label": label,
                "p6_clip": clip,
                "m6_machine_id": "id_00",
                "m6_label": m6_label,
                "m6_clip": m6_clip,
                "kept": True,
            }
        )
    return pd.DataFrame(rows)


# At -6 dB, normal recording 3 is stored as abnormal 00000001 and vice versa;
# normal recordings 0 and 1 swap names.
M6 = [
    ("normal", "00000001"),
    ("normal", "00000000"),
    ("normal", "00000002"),
    ("abnormal", "00000001"),
    ("abnormal", "00000000"),
    ("normal", "00000003"),
]


def test_identity_files_use_the_matched_file_and_the_reference_label():
    files = identity_files(_identities(M6))
    check_content_identity(files)
    m6 = files[files["snr_db"] == -6].set_index(["label", "clip"])
    assert m6.loc[("normal", "00000003"), "file_label"] == "abnormal"
    assert m6.loc[("normal", "00000003"), "file_clip"] == "00000001"
    assert m6.loc[("abnormal", "00000001"), "file_label"] == "normal"
    # Labels used for evaluation are those of the identities (0 dB references).
    assert sorted(files["label"].unique()) == ["abnormal", "normal"]


def test_excluded_identities_have_no_file():
    identities = _identities(M6)
    identities.loc[0, "kept"] = False
    files = identity_files(identities)
    assert len(files) == 3 * 5
    assert "00000000" not in set(files.loc[files["label"] == "normal", "clip"])


def test_content_split_has_no_leak_while_a_name_split_would():
    identities = _identities(M6)
    split = identities.loc[:, list(IDENTITY_COLUMNS)].copy()
    split["split"] = [TRAIN, TEST, TRAIN, TEST, TEST, TEST]
    check_split(split)
    files = assign_files(split, identity_files(identities))
    check_no_leakage(files)

    # Recording held by each file, keyed by (SNR, file label, file clip).
    holder = files.set_index(["snr_db", "file_label", "file_clip"])["split"]
    # Content-based: training identities use files holding training recordings.
    train = files[files["split"] == TRAIN]
    held = [holder[(r.snr_db, r.file_label, r.file_clip)] for r in train.itertuples()]
    assert set(held) == {TRAIN}
    # Name-based: at -6 dB the file named normal/00000000 (a training name)
    # holds normal recording 1, which is in the test set.
    assert holder[(-6, "normal", "00000000")] == TEST


def test_shared_file_is_rejected():
    identities = _identities(M6)
    identities.loc[1, ["m6_label", "m6_clip"]] = ["normal", "00000001"]
    with pytest.raises(ValueError, match="shared"):
        check_content_identity(identity_files(identities))


def test_missing_snr_file_is_rejected():
    files = identity_files(_identities(M6))
    files = files.drop(index=files.index[files["snr_db"] == 6][:1])
    with pytest.raises(ValueError, match="exactly one file per SNR"):
        check_content_identity(files)
