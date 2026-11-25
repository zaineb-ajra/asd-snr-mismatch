import hashlib
import re
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest

from noisemismatch.data.download import (
    Archive,
    aria2c_input,
    download_urllib,
    load_manifest,
    md5sum,
    verify_archive,
)


def test_manifest_covers_the_full_grid():
    archives = load_manifest()
    assert len(archives) == 12
    grid = {(a.snr_db, a.machine_type) for a in archives}
    assert grid == {
        (snr, mt) for snr in (-6, 0, 6) for mt in ("fan", "pump", "slider", "valve")
    }
    for a in archives:
        assert a.filename == f"{a.snr_db}_dB_{a.machine_type}.zip"
        assert re.fullmatch(r"[0-9a-f]{32}", a.md5)
    # Total reported by the Zenodo API for record 3384388.
    assert sum(a.size_bytes for a in archives) == 100_243_834_344


def _archive_for(path, data: bytes) -> Archive:
    return Archive(path.name, 0, "fan", len(data), hashlib.md5(data).hexdigest())


def test_md5sum_matches_hashlib_across_chunks(tmp_path):
    data = bytes(range(256)) * 1000
    path = tmp_path / "f.bin"
    path.write_bytes(data)
    assert md5sum(path, chunk_size=1000) == hashlib.md5(data).hexdigest()


def test_verify_archive_ok(tmp_path):
    data = b"mimii" * 100
    path = tmp_path / "a.zip"
    path.write_bytes(data)
    result = verify_archive(path, _archive_for(path, data))
    assert result.ok
    assert result.reason == "ok"


def test_verify_archive_missing(tmp_path):
    path = tmp_path / "a.zip"
    result = verify_archive(path, _archive_for(path, b"x"))
    assert not result.ok
    assert result.reason == "missing"


def test_verify_archive_wrong_size_is_not_hashed(tmp_path):
    path = tmp_path / "a.zip"
    path.write_bytes(b"short")
    result = verify_archive(path, _archive_for(path, b"longer content"))
    assert not result.ok
    assert result.md5 is None


def test_verify_archive_wrong_md5(tmp_path):
    path = tmp_path / "a.zip"
    path.write_bytes(b"abcd")
    result = verify_archive(path, _archive_for(path, b"abce"))
    assert not result.ok
    assert result.reason.startswith("md5")


def test_verify_archive_unfinished_aria2c_download(tmp_path):
    data = b"abcd"
    path = tmp_path / "a.zip"
    path.write_bytes(data)
    (tmp_path / "a.zip.aria2").write_bytes(b"")
    result = verify_archive(path, _archive_for(path, data))
    assert not result.ok
    assert result.reason == "incomplete download"


def test_aria2c_input_lists_url_name_and_checksum():
    archive = Archive("-6_dB_fan.zip", -6, "fan", 1, "0" * 32)
    text = aria2c_input([archive])
    assert text.splitlines() == [
        "https://zenodo.org/api/records/3384388/files/-6_dB_fan.zip/content",
        "  out=-6_dB_fan.zip",
        "  checksum=md5=" + "0" * 32,
    ]


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


@pytest.fixture
def http_dir(tmp_path):
    served = tmp_path / "served"
    served.mkdir()
    handler = partial(_QuietHandler, directory=str(served))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield served, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_urllib_fallback_downloads_and_renames(http_dir, tmp_path):
    served, base_url = http_dir
    data = bytes(range(256)) * 5000
    (served / "a.zip").write_bytes(data)
    dest = tmp_path / "a.zip"
    download_urllib(f"{base_url}/a.zip", dest)
    assert dest.read_bytes() == data
    assert not (tmp_path / "a.zip.part").exists()
