import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from noisemismatch.data.dcase2020 import DCASE_NAME, mono_wav_bytes
from noisemismatch.data.remote_zip import open_remote_zip


def _server(payload: bytes):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()

        def do_GET(self):
            start, end = self.headers["Range"].removeprefix("bytes=").split("-")
            chunk = payload[int(start) : int(end) + 1]
            self.send_response(206)
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def remote_zip(tmp_path):
    path = tmp_path / "a.zip"
    rng = np.random.default_rng(0)
    members = {k: rng.bytes(50_000) for k in range(20)}  # incompressible, like audio
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for k, data in members.items():
            zf.writestr(f"fan/train/normal_id_00_{k:08d}.wav", data)
    server = _server(path.read_bytes())
    yield f"http://127.0.0.1:{server.server_address[1]}/a.zip", path, members
    server.shutdown()
    server.server_close()


def test_remote_listing_and_member_without_full_download(remote_zip):
    url, path, members = remote_zip
    reader, raw = open_remote_zip(url)
    with zipfile.ZipFile(reader) as zf, zipfile.ZipFile(path) as local:
        assert [i.CRC for i in zf.infolist()] == [i.CRC for i in local.infolist()]
        listed = raw.downloaded
        assert zf.read("fan/train/normal_id_00_00000007.wav") == members[7]
    assert listed < path.stat().st_size / 4  # the listing only reads the end


def test_mono_wav_bytes_has_a_44_byte_header():
    x = np.arange(-50, 50, dtype=np.int16)
    data = mono_wav_bytes(x)
    assert len(data) == 44 + 2 * len(x)
    assert data[:4] == b"RIFF"
    assert data[36:40] == b"data"
    assert np.array_equal(np.frombuffer(data[44:], dtype="<i2"), x)


def test_dcase_name_pattern():
    match = DCASE_NAME.fullmatch("valve/test/anomaly_id_02_00000005.wav")
    assert match.groupdict() == {
        "machine_type": "valve",
        "dcase_split": "test",
        "dcase_label": "anomaly",
        "machine_id": "id_02",
        "dcase_clip": "00000005",
    }


def test_other_snrs_check_counts_matches(tmp_path, monkeypatch):
    import io
    import zlib

    import pandas as pd
    import soundfile as sf

    from noisemismatch.data import dcase2020

    rng = np.random.default_rng(0)
    waves = {k: rng.integers(-3000, 3000, (160000, 8), dtype=np.int16) for k in "ab"}
    with zipfile.ZipFile(tmp_path / "0_dB_fan.zip", "w") as zf:
        for k, wave in waves.items():
            buffer = io.BytesIO()
            sf.write(buffer, wave, 16000, format="WAV", subtype="PCM_16")
            zf.writestr(
                f"fan/id_00/normal/0000000{int(k == 'b')}.wav", buffer.getvalue()
            )
    # the DCASE side holds channel 0 of file "a" only
    mapping = tmp_path / "mapping.csv"
    pd.DataFrame(
        {
            "machine_type": ["fan"],
            "dcase_member": ["fan/train/normal_id_00_00000000.wav"],
            "crc32": [f"{zlib.crc32(mono_wav_bytes(waves['a'][:, 0])):08x}"],
        }
    ).to_csv(mapping, index=False)
    monkeypatch.setattr(dcase2020, "MAPPING_PATH", mapping)
    monkeypatch.setattr(dcase2020, "MACHINE_TYPES", ("fan",))
    index = pd.DataFrame(
        {
            "machine_type": "fan",
            "snr_db": 0,
            "archive": "0_dB_fan.zip",
            "member": [
                "fan/id_00/normal/00000000.wav",
                "fan/id_00/normal/00000001.wav",
            ],
        }
    )
    check = dcase2020.other_snrs_check(index, tmp_path, workers=2, snrs=(0,))
    assert check.loc[0, "n_mimii_files"] == 2
    assert check.loc[0, "n_matching_dcase"] == 1
    assert check.loc[0, "matching_members"] == "fan/id_00/normal/00000000.wav"
