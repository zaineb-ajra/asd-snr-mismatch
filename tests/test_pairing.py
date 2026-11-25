import io
import zipfile

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from noisemismatch.data.download import Archive
from noisemismatch.data.mimii import build_file_index
from noisemismatch.data.pairing import content_pairing, unpaired_identities
from noisemismatch.data.prepare import inventory, split_summary
from noisemismatch.data.split import assign_files, make_split

SNRS = (-6, 0, 6)
FRAMES = 4000


def _clips(n_normal=4, n_abnormal=3):
    return [("normal", f"{i:08d}") for i in range(n_normal)] + [
        ("abnormal", f"{i:08d}") for i in range(n_abnormal)
    ]


def _build(tmp_path, shared_noise: bool):
    """Writes three fan archives built as machine + gain(SNR) * noise."""
    rng = np.random.default_rng(0)
    clips = _clips()
    machine = {c: rng.normal(size=FRAMES) for c in clips}
    noise = {c: rng.normal(size=FRAMES) for c in clips}
    archives = []
    for snr in SNRS:
        archive = Archive(f"{snr}_dB_fan.zip", snr, "fan", 0, "")
        archives.append(archive)
        gain = 10 ** (-snr / 20)
        with zipfile.ZipFile(tmp_path / archive.filename, "w") as zf:
            for label, clip in clips:
                n = noise[(label, clip)] if shared_noise else rng.normal(size=FRAMES)
                x = (machine[(label, clip)] + gain * n) / 20
                buffer = io.BytesIO()
                sf.write(buffer, np.stack([x, x], axis=1), 16000, format="WAVEX")
                zf.writestr(f"fan/id_00/{label}/{clip}.wav", buffer.getvalue())
    return build_file_index(tmp_path, archives)


def test_unpaired_identities():
    rows = [("fan", "id_00", "normal", "00000000", snr) for snr in SNRS] + [
        ("fan", "id_00", "normal", "00000001", 0)
    ]
    index = pd.DataFrame(
        rows, columns=["machine_type", "machine_id", "label", "clip", "snr_db"]
    )
    unpaired = unpaired_identities(index)
    assert list(unpaired["clip"]) == ["00000001"]
    assert unpaired["snrs_present"].iloc[0] == (0,)


@pytest.mark.parametrize("shared_noise", [True, False])
def test_content_pairing_separates_same_and_different_clips(tmp_path, shared_noise):
    index = _build(tmp_path, shared_noise)
    result = content_pairing(index, tmp_path, per_group=3, seed=0)
    assert len(result) == 6  # 3 normal + 3 abnormal clips drawn
    assert (result["corr_0_6"] > 0.5).all()
    assert (result["corr_other"].abs() < 0.1).all()
    if shared_noise:
        # Same noise, rescaled: the -6 dB version is in the span of the others,
        # up to 16-bit quantization.
        assert (result["residual"] < 0.01).all()
    else:
        assert (result["residual"] > 0.3).all()


def test_inventory_and_split_summary(tmp_path):
    index = _build(tmp_path, shared_noise=True)
    table = inventory(index)
    assert len(table) == 2 * len(SNRS)  # (normal, abnormal) x SNR
    normal = table[table["label"] == "normal"]
    assert (normal["n_files"] == 4).all()
    assert (table["channels"] == 2).all()
    assert (table["frames"] == FRAMES).all()

    identities = index[["machine_type", "machine_id", "label", "clip"]]
    files = assign_files(make_split(identities.drop_duplicates(), 0), index)
    summary = split_summary(files)
    assert summary[
        ["train_normal", "test_normal", "test_abnormal"]
    ].values.tolist() == [[1, 3, 3]]
