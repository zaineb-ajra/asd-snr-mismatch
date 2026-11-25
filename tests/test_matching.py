import io
import zipfile

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from noisemismatch.data.download import Archive
from noisemismatch.data.identity import (
    MatchRule,
    identity_table,
    match_identities,
    symmetry_table,
)
from noisemismatch.data.matching import (
    block_scores,
    compute_type,
    cross_type_screen,
    standardize,
    type_files,
)
from noisemismatch.data.mimii import build_file_index

CPU = torch.device("cpu")
FRAMES = 4000
CHANNELS = 2


def test_standardize_and_block_scores_give_mean_channel_correlation():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, FRAMES, CHANNELS))
    b = a[[2, 0]] + rng.normal(size=(2, FRAMES, CHANNELS))
    za = np.stack([standardize(x) for x in a]).astype(np.float16)
    zb = np.stack([standardize(x) for x in b]).astype(np.float16)
    scores = block_scores(za.reshape(3, -1), zb.reshape(2, -1), CPU, block=2)
    expected = np.array(
        [
            [
                np.mean([np.corrcoef(a[i, :, c], b[j, :, c])[0, 1] for c in range(2)])
                for j in range(2)
            ]
            for i in range(3)
        ]
    )
    assert np.allclose(scores, expected, atol=2e-3)


def test_standardize_constant_channel_is_zero():
    x = np.zeros((10, 2), dtype=np.float32)
    x[:, 1] = np.arange(10)
    z = standardize(x)
    assert np.all(z[0] == 0)
    assert abs(z[1].mean()) < 1e-6
    assert abs(z[1].std() - 1) < 1e-5


def _write_type(tmp_path, machine_type, recordings, layout, seed):
    """Writes one archive per SNR; layout[snr] maps member name -> recording key."""
    rng = np.random.default_rng(seed)
    archives = []
    for snr, members in layout.items():
        archive = Archive(f"{snr}_dB_{machine_type}.zip", snr, machine_type, 0, "")
        archives.append(archive)
        gain = 10 ** (-snr / 20)
        with zipfile.ZipFile(tmp_path / archive.filename, "w") as zf:
            for member, key in members.items():
                x = (recordings[key] + gain * rng.normal(size=(FRAMES, CHANNELS))) / 20
                buffer = io.BytesIO()
                sf.write(buffer, x, 16000, format="WAVEX")
                zf.writestr(f"{machine_type}/{member}", buffer.getvalue())
    return archives


def test_permutation_across_labels_is_recovered(tmp_path):
    rng = np.random.default_rng(1)
    keys = ["n0", "n1", "n2", "a0", "a1"]
    recordings = {k: rng.normal(size=(FRAMES, CHANNELS)) for k in keys}
    faithful = {
        "id_00/normal/00000000.wav": "n0",
        "id_00/normal/00000001.wav": "n1",
        "id_00/normal/00000002.wav": "n2",
        "id_00/abnormal/00000000.wav": "a0",
        "id_00/abnormal/00000001.wav": "a1",
    }
    # At -6 dB the normal recording n2 sits in the abnormal folder and vice versa.
    swapped = dict(faithful)
    swapped["id_00/normal/00000002.wav"] = "a1"
    swapped["id_00/abnormal/00000001.wav"] = "n2"
    layout = {0: faithful, 6: faithful, -6: swapped}
    archives = _write_type(tmp_path, "fan", recordings, layout, seed=2)
    index = build_file_index(tmp_path, archives)
    ch0 = compute_type(index, tmp_path, "fan", tmp_path, CPU, workers=2)
    assert set(ch0) == {0, 6, -6}

    scores = np.load(tmp_path / "fan_scores.npz")
    rule = MatchRule(min_score=0.10, min_margin=0.05, min_support=0.05)
    result = match_identities(scores["s_p6"], scores["s_0m"], scores["s_6m"], rule)
    assert result["kept"].all()
    refs = type_files(index, "fan", 0)
    m6 = type_files(index, "fan", -6).loc[result["m6_col"]].reset_index(drop=True)
    crossed = (refs["label"] != m6["label"]).to_numpy()
    # Exactly the two swapped recordings cross the label boundary.
    assert refs.loc[crossed, "clip"].tolist() == ["00000001", "00000002"]
    assert refs.loc[crossed, "label"].tolist() == ["abnormal", "normal"]


def test_cross_type_screen_prefers_own_type(tmp_path):
    rng = np.random.default_rng(3)
    ch0 = {}
    frames = []
    for t in ("fan", "pump"):
        recordings = {k: rng.normal(size=(FRAMES, CHANNELS)) for k in ("r0", "r1")}
        members = {
            "id_00/normal/00000000.wav": "r0",
            "id_00/normal/00000001.wav": "r1",
        }
        archives = _write_type(
            tmp_path, t, recordings, {s: members for s in (0, 6, -6)}, 4
        )
        frames.append(build_file_index(tmp_path, archives))
    index = pd.concat(frames, ignore_index=True)
    for t in ("fan", "pump"):
        ch0[t] = compute_type(index, tmp_path, t, tmp_path, CPU, workers=2)
    screen = cross_type_screen(ch0, index, CPU)
    assert len(screen) == 8  # 2 types x 2 files x 2 compared SNRs
    assert (screen["best_own_type"] > screen["best_other_type"] + 0.1).all()


def test_symmetry_table_on_a_permutation(tmp_path):
    rng = np.random.default_rng(5)
    keys = [f"n{i}" for i in range(6)]
    recordings = {k: rng.normal(size=(FRAMES, CHANNELS)) for k in keys}
    faithful = {f"id_00/normal/{i:08d}.wav": f"n{i}" for i in range(6)}
    permuted = {f"id_00/normal/{i:08d}.wav": f"n{(i + 1) % 6}" for i in range(6)}
    archives = _write_type(
        tmp_path, "fan", recordings, {0: faithful, 6: faithful, -6: permuted}, 6
    )
    index = build_file_index(tmp_path, archives)
    compute_type(index, tmp_path, "fan", tmp_path, CPU, workers=2)
    rule = MatchRule(min_score=0.10, min_margin=0.05, min_support=0.05)
    table = identity_table(tmp_path, "fan", rule)
    assert table["kept"].all()
    assert not table["m6_same_name"].any()
    symmetry = symmetry_table(table, tmp_path, rule).set_index("snr_db")
    assert symmetry.loc[-6, "accepted"] == 6
    assert symmetry.loc[-6, "other_name"] == 6
    assert symmetry.loc[6, "other_name"] == 0
    assert (symmetry["sign_exceeded"] == 0).all()
    assert (symmetry["p_binomial_half"] == 0.5**6).all()
    assert (symmetry["median_accepted"] > symmetry["null_p999_abs"]).all()
