import numpy as np
import pandas as pd
import pytest

from noisemismatch.data.dataset import (
    ClipSet,
    build_clip_table,
    select,
    strict_exclusion,
)

FRAMES = 100
STRICT = {
    "exclude_if_false": ["m6_argmax_agree"],
    "exclude_if_true": ["m6_below_null_p999"],
}


def _identities():
    rows = []
    for machine_type, machine_id in (("fan", "id_00"), ("pump", "id_02")):
        for label, clips in (("normal", range(4)), ("abnormal", range(2))):
            for c in clips:
                clip = f"{c:08d}"
                # At -6 dB the files of clips 0 and 1 are swapped (name permutation).
                m6 = f"{(1 - c) if c < 2 else c:08d}"
                rows.append(
                    {
                        "machine_type": machine_type,
                        "machine_id": machine_id,
                        "label": label,
                        "clip": clip,
                        "p6_machine_id": machine_id,
                        "p6_label": label,
                        "p6_clip": clip,
                        "m6_machine_id": machine_id,
                        "m6_label": label,
                        "m6_clip": m6,
                        "kept": True,
                        "m6_argmax_agree": c != 3,
                        "m6_below_null_p999": False,
                    }
                )
    return pd.DataFrame(rows)


def _split(identities):
    split = identities[["machine_type", "machine_id", "label", "clip"]].copy()
    split["split"] = np.where(
        (split["label"] == "abnormal") | split["clip"].isin(["00000000", "00000001"]),
        "test",
        "train",
    )
    return split


class FakeAudio:
    """Waveform of a file = constant equal to a code of (type, snr, file)."""

    fingerprint = "fake"

    @staticmethod
    def code(machine_type, snr, machine_id, label, clip):
        return (hash((machine_type, snr, machine_id, label, clip)) % 20000) - 10000

    def load(self, machine_type, snr, files):
        return np.stack(
            [
                np.full(FRAMES, self.code(machine_type, snr, *r), dtype=np.int16)
                for r in files.to_numpy()
            ]
        )


@pytest.fixture
def table():
    identities = _identities()
    return build_clip_table(_split(identities), identities, (0, 6, -6), STRICT)


def test_clip_table_columns_order_and_labels(table):
    assert len(table) == 12 * 3
    ordered = table.sort_values(
        ["machine_type", "machine_id", "label", "clip", "snr_db"]
    )
    assert table.index.equals(ordered.index)
    assert set(table.loc[table["label"] == "abnormal", "y"]) == {1}
    assert set(table.loc[table["label"] == "normal", "y"]) == {0}


def test_files_follow_content_identity(table):
    row = table[
        (table["machine_type"] == "fan")
        & (table["label"] == "normal")
        & (table["clip"] == "00000000")
        & (table["snr_db"] == -6)
    ].iloc[0]
    assert row["file_clip"] == "00000001"


def test_strict_flags(table):
    flagged = table.loc[table["strict_exclude"], "clip"].unique().tolist()
    assert flagged == ["00000003"]
    identities = _identities()
    identities.loc[0, "m6_below_null_p999"] = True
    assert strict_exclusion(identities, STRICT).sum() == 3  # 2 argmax + 1 percentile


def test_select_keeps_order(table):
    train = select(table, partition="train", snrs=(6,))
    assert (train["split"] == "train").all()
    assert (train["snr_db"] == 6).all()
    assert train["clip"].tolist() == ["00000002", "00000003"] * 2
    multi = select(table, partition="train", snrs=(0, 6, -6), machine_types=("fan",))
    assert len(multi) == 2 * 3


def test_waveforms_are_reassembled_in_table_order(table):
    clips = ClipSet(select(table, snrs=(0, -6)), FakeAudio())
    waveforms = clips.waveforms()
    assert waveforms.shape == (len(clips), FRAMES)
    assert waveforms.dtype == np.float32
    for k, row in clips.table.iterrows():
        code = FakeAudio.code(
            row["machine_type"],
            row["snr_db"],
            row["file_machine_id"],
            row["file_label"],
            row["file_clip"],
        )
        assert waveforms[k, 0] == pytest.approx(code / 32768)


def test_groups_by_scope(table):
    clips = ClipSet(table, FakeAudio())
    assert [k for k, _ in clips.groups("machine_id")] == [
        ("fan", "id_00"),
        ("pump", "id_02"),
    ]
    assert [k for k, _ in clips.groups("machine_type")] == [("fan",), ("pump",)]
    assert [len(c) for _, c in clips.groups("all")] == [len(clips)]
