import io
import zipfile

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from noisemismatch.config import load_paths
from noisemismatch.data.audio_cache import (
    FRAMES,
    AudioCache,
    build_cache,
    cache_dir,
    member_name,
    read_channel,
)
from noisemismatch.data.download import DOWNLOAD_SUBDIR

FILES = [("id_00", "normal", "00000000"), ("id_00", "abnormal", "00000003")]


def _write_archive(path, machine_type, rng):
    samples = {}
    with zipfile.ZipFile(path, "w") as zf:
        for machine_id, label, clip in FILES:
            x = rng.integers(-3000, 3000, size=(FRAMES, 8), dtype=np.int16)
            buffer = io.BytesIO()
            sf.write(buffer, x, 16000, format="WAVEX", subtype="PCM_16")
            zf.writestr(
                member_name(machine_type, machine_id, label, clip), buffer.getvalue()
            )
            samples[(machine_id, label, clip)] = x[:, 0]
    return samples


def test_cache_holds_channel_zero_bit_for_bit(tmp_path):
    rng = np.random.default_rng(0)
    expected = {}
    for snr in (0, 6):
        expected[snr] = _write_archive(tmp_path / f"{snr}_dB_fan.zip", "fan", rng)
    files = pd.DataFrame(
        [("fan", snr, *f) for snr in (0, 6) for f in FILES],
        columns=[
            "machine_type",
            "snr_db",
            "file_machine_id",
            "file_label",
            "file_clip",
        ],
    )
    root = tmp_path / "cache"
    manifest = build_cache(files, tmp_path, root, workers=2)
    assert manifest["channel"] == 0
    assert "fan_0" in manifest["fingerprint"]

    cache = AudioCache(root)
    request = files[files["snr_db"] == 6].iloc[::-1]  # any order
    loaded = cache.load("fan", 6, request)
    assert loaded.dtype == np.int16
    for row, key in zip(
        loaded, map(tuple, request.iloc[:, 2:].to_numpy()), strict=True
    ):
        assert np.array_equal(row, expected[6][key])
    with pytest.raises(FileExistsError):
        build_cache(files, tmp_path, root, workers=2)


@pytest.mark.data
def test_real_cache_matches_the_archives():
    """Compares 24 random cached waveforms with channel 0 read from the archives."""
    paths = load_paths()
    root = cache_dir(paths.cache_root)
    if not (root / "manifest.json").exists():
        pytest.skip("audio cache not built")
    cache = AudioCache(root)
    rng = np.random.default_rng(2026)
    for entry in rng.permutation(cache.manifest["entries"])[:12]:
        machine_type, snr = entry["machine_type"], entry["snr_db"]
        index = pd.read_csv(root / f"{machine_type}_{snr}.csv", dtype=str)
        sample = index.iloc[rng.choice(len(index), size=2, replace=False)]
        cached = cache.load(machine_type, snr, sample)
        with zipfile.ZipFile(
            paths.data_root / DOWNLOAD_SUBDIR / entry["archive"]
        ) as zf:
            for row, (machine_id, label, clip) in zip(
                cached, sample.to_numpy(), strict=True
            ):
                member = member_name(machine_type, machine_id, label, clip)
                assert np.array_equal(row, read_channel(zf, member))
