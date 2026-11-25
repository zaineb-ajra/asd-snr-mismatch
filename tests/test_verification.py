import io
import zipfile

import numpy as np
import pandas as pd
import soundfile as sf

from noisemismatch.data.audio_cache import FRAMES, member_name
from noisemismatch.data.split import TEST, TRAIN
from noisemismatch.verification import ArchiveAudio, original_split_table


def _index():
    rows = [("pump", "id_02", "normal", f"{k:08d}", 6) for k in (5, 0, 3, 1, 4, 2)]
    rows += [("pump", "id_02", "abnormal", f"{k:08d}", 6) for k in (1, 0)]
    rows += [("pump", "id_02", "normal", "00000000", 0)]  # other SNR, ignored
    rows += [("pump", "id_04", "normal", "00000000", 6)]  # other ID, ignored
    return pd.DataFrame(
        rows, columns=["machine_type", "machine_id", "label", "clip", "snr_db"]
    )


def test_original_split_puts_the_first_sorted_normal_files_in_test():
    table = original_split_table(_index(), "pump", "id_02", 6)
    normal = table[table["label"] == "normal"].set_index("clip")["split"]
    # Two anomalous files -> the first two normal files by name are test files.
    assert normal.to_dict() == {
        "00000000": TEST,
        "00000001": TEST,
        "00000002": TRAIN,
        "00000003": TRAIN,
        "00000004": TRAIN,
        "00000005": TRAIN,
    }
    assert (table.loc[table["label"] == "abnormal", "split"] == TEST).all()
    assert (table["file_clip"] == table["clip"]).all()  # names at face value
    assert not table["strict_exclude"].any()


def test_archive_audio_reads_channel_zero(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    files = pd.DataFrame(
        {
            "file_machine_id": ["id_02"] * 2,
            "file_label": ["normal"] * 2,
            "file_clip": ["00000000", "00000001"],
        }
    )
    expected = {}
    with zipfile.ZipFile(tmp_path / "6_dB_pump.zip", "w") as zf:
        for clip in files["file_clip"]:
            x = rng.integers(-500, 500, size=(FRAMES, 8), dtype=np.int16)
            buffer = io.BytesIO()
            sf.write(buffer, x, 16000, format="WAVEX", subtype="PCM_16")
            zf.writestr(member_name("pump", "id_02", "normal", clip), buffer.getvalue())
            expected[clip] = x[:, 0]
    audio = ArchiveAudio(tmp_path, "pump", 6, "id_02", files)
    assert "6_dB_pump.zip" in audio.fingerprint
    loaded = audio.load("pump", 6, files.iloc[::-1])
    assert np.array_equal(loaded[0], expected["00000001"])
    assert np.array_equal(loaded[1], expected["00000000"])
