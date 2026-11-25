"""Cache of channel-0 waveforms of the files used by the splits.

All models read channel 0 (microphone 1, as the MIMII baseline). The cache holds,
for each (machine type, SNR), the raw 16-bit samples of channel 0 of every file
used by the main or the sensitivity split, in one ``.npy`` array (shape
``(n_files, frames)``, int16) with an index CSV giving the file of each row. Samples
are copied bit for bit from the archives; a test compares a random sample with the
archives.

Usage::

    python -m noisemismatch.data.audio_cache [--workers 12]

Layout under ``<cache_root>/audio/<CACHE_NAME>/``: ``<type>_<snr>.npy``,
``<type>_<snr>.csv`` and ``manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from noisemismatch.config import CONFIG_DIR, load_paths, load_yaml
from noisemismatch.data.build_split import split_file_tables
from noisemismatch.data.download import DOWNLOAD_SUBDIR, LOG_SUBDIR, load_manifest
from noisemismatch.data.identity import load_identity
from noisemismatch.data.split import FILE_COLUMNS
from noisemismatch.utils.logs import setup_logging
from noisemismatch.utils.provenance import git_commit

logger = logging.getLogger(__name__)

CACHE_NAME: str = "ch0-v1"
CHANNEL: int = 0
SAMPLE_RATE: int = 16_000
FRAMES: int = 160_000
FILE_KEY: list[str] = ["machine_type", *FILE_COLUMNS]


def cache_dir(cache_root: Path) -> Path:
    """Directory of the audio cache under ``cache_root``."""
    return cache_root / "audio" / CACHE_NAME


def member_name(machine_type: str, machine_id: str, label: str, clip: str) -> str:
    """Archive member of a file."""
    return f"{machine_type}/{machine_id}/{label}/{clip}.wav"


def read_channel(archive: zipfile.ZipFile, member: str) -> np.ndarray:
    """Reads the int16 samples of channel 0 of one archive member."""
    with archive.open(member) as f:
        samples, rate = sf.read(f, dtype="int16", always_2d=True)
    if rate != SAMPLE_RATE or samples.shape[0] != FRAMES:
        raise ValueError(f"{member}: unexpected rate {rate} or length {len(samples)}")
    return samples[:, CHANNEL]


def build_array(archive_path: Path, files: pd.DataFrame, workers: int) -> np.ndarray:
    """Reads channel 0 of the given files of one archive, in row order.

    Args:
        archive_path: Archive (one machine type, one SNR).
        files: Rows with ``machine_type`` and the file columns.
        workers: Reading threads (one archive handle per thread).

    Returns:
        Int16 array shaped ``(len(files), FRAMES)``.
    """
    out = np.empty((len(files), FRAMES), dtype=np.int16)
    local = threading.local()
    opened: list[zipfile.ZipFile] = []
    lock = threading.Lock()

    def load(k: int, member: str) -> None:
        if not hasattr(local, "archive"):
            local.archive = zipfile.ZipFile(archive_path)
            with lock:
                opened.append(local.archive)
        out[k] = read_channel(local.archive, member)

    members = [member_name(*row) for row in files[FILE_KEY].itertuples(index=False)]
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda job: load(*job), enumerate(members)))
    finally:
        for archive in opened:
            archive.close()
    return out


class AudioCache:
    """Read access to the channel-0 cache.

    Args:
        root: Cache directory (:func:`cache_dir`).
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"no audio cache at {root}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._index: dict[tuple[str, int], dict[tuple[str, str, str], int]] = {}

    @property
    def fingerprint(self) -> str:
        """Identifier of the cache content, used in feature-cache keys."""
        return str(self.manifest["fingerprint"])

    def _rows(self, machine_type: str, snr: int) -> dict[tuple[str, str, str], int]:
        key = (machine_type, snr)
        if key not in self._index:
            index = pd.read_csv(self.root / f"{machine_type}_{snr}.csv", dtype=str)
            self._index[key] = {
                tuple(r): n for n, r in enumerate(index[list(FILE_COLUMNS)].to_numpy())
            }
        return self._index[key]

    def index(self, machine_type: str, snr: int) -> pd.DataFrame:
        """File columns of the rows of one cached array, in row order."""
        return pd.read_csv(self.root / f"{machine_type}_{snr}.csv", dtype=str)

    def array(self, machine_type: str, snr: int) -> np.ndarray:
        """Memory-mapped int16 array of one machine type and SNR."""
        return np.load(self.root / f"{machine_type}_{snr}.npy", mmap_mode="r")

    def load(self, machine_type: str, snr: int, files: pd.DataFrame) -> np.ndarray:
        """Returns the int16 waveforms of the given files of one type and SNR.

        Args:
            machine_type: Machine type.
            snr: SNR in dB.
            files: Rows with the file columns, in the order to return.

        Returns:
            Int16 array shaped ``(len(files), FRAMES)``.
        """
        rows = self._rows(machine_type, snr)
        order = [rows[tuple(r)] for r in files[list(FILE_COLUMNS)].to_numpy()]
        array = np.load(self.root / f"{machine_type}_{snr}.npy", mmap_mode="r")
        return np.asarray(array[order])


def build_cache(
    files: pd.DataFrame, downloads_dir: Path, root: Path, workers: int
) -> dict:
    """Writes the cache for the given files (all machine types and SNRs).

    Args:
        files: Rows with ``machine_type``, ``snr_db`` and the file columns.
        downloads_dir: Directory holding the archives.
        root: Cache directory; must not contain a manifest yet.
        workers: Reading threads.

    Returns:
        The manifest written to ``root/manifest.json``.
    """
    if (root / "manifest.json").exists():
        raise FileExistsError(f"audio cache already built at {root}")
    root.mkdir(parents=True, exist_ok=True)
    md5 = {a.filename: a.md5 for a in load_manifest()}
    entries = []
    unique = files.drop_duplicates(["snr_db", *FILE_KEY])
    for (machine_type, snr), part in unique.groupby(["machine_type", "snr_db"]):
        part = part.sort_values(list(FILE_COLUMNS), ignore_index=True)
        archive = f"{snr}_dB_{machine_type}.zip"
        array = build_array(downloads_dir / archive, part, workers)
        stem = f"{machine_type}_{snr}"
        part[list(FILE_COLUMNS)].to_csv(
            root / f"{stem}.csv", index=False, lineterminator="\n"
        )
        tmp = root / f"{stem}.tmp.npy"
        np.save(tmp, array)
        tmp.replace(root / f"{stem}.npy")
        entries.append(
            {
                "machine_type": machine_type,
                "snr_db": int(snr),
                "files": len(part),
                "archive": archive,
                "archive_md5": md5[archive],
            }
        )
        logger.info("%s: %d files cached", stem, len(part))
    fingerprint = f"{CACHE_NAME}:" + ",".join(
        f"{e['machine_type']}_{e['snr_db']}:{e['files']}:{e['archive_md5']}"
        for e in entries
    )
    manifest = {
        "name": CACHE_NAME,
        "channel": CHANNEL,
        "sample_rate": SAMPLE_RATE,
        "frames": FRAMES,
        "dtype": "int16",
        "entries": entries,
        "fingerprint": fingerprint,
        "git_commit": git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Builds the channel-0 cache of the files of the saved splits.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code (0).
    """
    parser = argparse.ArgumentParser(description="Build the channel-0 audio cache.")
    parser.add_argument("--workers", type=int, default=12, help="Reading threads.")
    args = parser.parse_args(argv)
    paths = load_paths()
    setup_logging(paths.data_root / LOG_SUBDIR / "audio_cache.log")
    protocol = load_yaml(CONFIG_DIR / "protocol.yaml")
    tables = split_file_tables(load_identity(), protocol)
    files = pd.concat(list(tables.values()), ignore_index=True)
    manifest = build_cache(
        files,
        paths.data_root / DOWNLOAD_SUBDIR,
        cache_dir(paths.cache_root),
        args.workers,
    )
    logger.info("Audio cache built: %d arrays", len(manifest["entries"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
