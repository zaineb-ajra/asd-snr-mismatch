import io
import zipfile

import numpy as np
import pytest
import soundfile as sf

from noisemismatch.data.download import Archive
from noisemismatch.data.mimii import (
    build_file_index,
    index_archive,
    parse_member_name,
    parse_wav_header,
    read_member,
)

SR = 16000


def _wav_bytes(samples: np.ndarray, fmt: str) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, samples, SR, format=fmt, subtype="PCM_16")
    return buffer.getvalue()


def _samples(frames=1600, channels=8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.5, 0.5, size=(frames, channels))


@pytest.mark.parametrize(("fmt", "extensible"), [("WAV", False), ("WAVEX", True)])
def test_parse_wav_header_matches_soundfile(fmt, extensible):
    data = _wav_bytes(_samples(), fmt)
    header = parse_wav_header(data[:1024])
    info = sf.info(io.BytesIO(data))
    assert header.extensible is extensible
    assert header.format_tag == 1  # PCM
    assert header.channels == info.channels == 8
    assert header.sample_rate == info.samplerate == SR
    assert header.bits_per_sample == 16
    assert header.frames == info.frames == 1600
    assert header.data_offset + header.data_bytes == len(data)


def test_parse_wav_header_rejects_non_wav():
    with pytest.raises(ValueError, match="RIFF"):
        parse_wav_header(b"not a wav file at all")


def test_parse_member_name():
    assert parse_member_name("pump/id_00/abnormal/00000012.wav") == {
        "machine_type": "pump",
        "machine_id": "id_00",
        "label": "abnormal",
        "clip": "00000012",
    }
    for bad in ("pump/id_00/abnormal/12.wav", "pump/id_0/normal/00000000.wav", "x.wav"):
        with pytest.raises(ValueError, match="unexpected archive member"):
            parse_member_name(bad)


def _write_archive(path, machine_type, members, extra=()):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{machine_type}/", b"")
        for i, member in enumerate(members):
            zf.writestr(member, _wav_bytes(_samples(seed=i), "WAVEX"))
        for name in extra:
            zf.writestr(name, b"junk")


def test_index_archive_and_read_member(tmp_path):
    members = ["fan/id_00/normal/00000000.wav", "fan/id_00/abnormal/00000003.wav"]
    _write_archive(tmp_path / "6_dB_fan.zip", "fan", members)
    archive = Archive("6_dB_fan.zip", 6, "fan", 0, "")
    index = index_archive(tmp_path / "6_dB_fan.zip", archive)
    assert list(index["member"]) == members
    assert list(index["label"]) == ["normal", "abnormal"]
    assert (index["snr_db"] == 6).all()
    assert (index["channels"] == 8).all()
    assert (index["duration_s"] == 0.1).all()

    x, sr = read_member(tmp_path / "6_dB_fan.zip", members[1], channel=2)
    expected = _samples(seed=1)[:, 2]
    assert sr == SR
    assert x.shape == (1600,)
    assert np.allclose(x, expected, atol=1 / 2**15)


def test_index_archive_rejects_foreign_members(tmp_path):
    _write_archive(tmp_path / "a.zip", "fan", ["pump/id_00/normal/00000000.wav"])
    with pytest.raises(ValueError, match="unexpected member"):
        index_archive(tmp_path / "a.zip", Archive("a.zip", 0, "fan", 0, ""))
    _write_archive(tmp_path / "b.zip", "fan", [], extra=["fan/readme.txt"])
    with pytest.raises(ValueError, match="unexpected archive member"):
        index_archive(tmp_path / "b.zip", Archive("b.zip", 0, "fan", 0, ""))


def test_build_file_index_sorts_across_archives(tmp_path):
    member = "fan/id_00/normal/00000000.wav"
    archives = [Archive(f"{snr}_dB_fan.zip", snr, "fan", 0, "") for snr in (6, -6)]
    for a in archives:
        _write_archive(tmp_path / a.filename, "fan", [member])
    index = build_file_index(tmp_path, archives)
    assert list(index["snr_db"]) == [-6, 6]
