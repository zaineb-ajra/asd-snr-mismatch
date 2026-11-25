"""Mapping of the DCASE 2020 Task 2 development files to MIMII files.

The MIMII part of the DCASE 2020 Task 2 development set (Zenodo 3678171; fan,
pump, slider, valve) corresponds to channel 0 of MIMII files at -6 dB, rewritten
as mono 16-bit WAV files under other names (docs/data.md). ToyCar and
ToyConveyor, the other machine types of that set, come from ToyADMOS. This module rebuilds that WAV file for every MIMII -6 dB
file, computes its CRC-32 and matches it with the CRC-32 listed in the central
directory of the DCASE archives, which are read through HTTP range requests (a few
MB; the DCASE data are not downloaded).

Usage::

    python -m noisemismatch.data.dcase2020 [--workers 12]

writes ``metadata/dcase2020_mimii_mapping.csv``.
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import threading
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from noisemismatch.config import load_paths
from noisemismatch.data.audio_cache import SAMPLE_RATE, read_channel
from noisemismatch.data.download import DOWNLOAD_SUBDIR, LOG_SUBDIR
from noisemismatch.data.prepare import METADATA_DIR, index_path, load_index
from noisemismatch.data.remote_zip import open_remote_zip
from noisemismatch.utils.logs import setup_logging

logger = logging.getLogger(__name__)

RECORD: int = 3678171
MACHINE_TYPES: tuple[str, ...] = ("fan", "pump", "slider", "valve")
MAPPING_PATH: Path = METADATA_DIR / "dcase2020_mimii_mapping.csv"
OTHER_SNRS_PATH: Path = METADATA_DIR / "dcase2020_other_snrs_check.csv"
DCASE_NAME: re.Pattern[str] = re.compile(
    r"(?P<machine_type>[a-z]+)/(?P<dcase_split>train|test)/"
    r"(?P<dcase_label>normal|anomaly)_(?P<machine_id>id_\d{2})_(?P<dcase_clip>\d{8})\.wav"
)


def mono_wav_bytes(samples: np.ndarray) -> bytes:
    """Mono 16-bit PCM WAV file (44-byte header) holding ``samples``."""
    buffer = io.BytesIO()
    sf.write(buffer, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def dcase_listing(machine_type: str) -> pd.DataFrame:
    """Members of one DCASE 2020 development archive with their CRC-32.

    Args:
        machine_type: MIMII machine type.

    Returns:
        One row per WAV member: parsed name fields, ``dcase_member`` and ``crc32``.
    """
    url = f"https://zenodo.org/api/records/{RECORD}/files/dev_data_{machine_type}.zip/content"
    reader, raw = open_remote_zip(url)
    with zipfile.ZipFile(reader) as zf:
        rows = []
        for info in zf.infolist():
            if info.is_dir():
                continue
            match = DCASE_NAME.fullmatch(info.filename)
            if match is None:
                raise ValueError(f"unexpected DCASE member {info.filename!r}")
            rows.append(
                {**match.groupdict(), "dcase_member": info.filename, "crc32": info.CRC}
            )
    logger.info(
        "%s: %d DCASE members listed (%d bytes read)",
        machine_type,
        len(rows),
        raw.downloaded,
    )
    return pd.DataFrame(rows)


def mimii_crcs(files: pd.DataFrame, archive: Path, workers: int) -> list[int]:
    """CRC-32 of the rebuilt mono WAV file of each MIMII file of one archive."""
    local = threading.local()
    opened: list[zipfile.ZipFile] = []
    lock = threading.Lock()

    def crc(member: str) -> int:
        if not hasattr(local, "zf"):
            local.zf = zipfile.ZipFile(archive)
            with lock:
                opened.append(local.zf)
        return zlib.crc32(mono_wav_bytes(read_channel(local.zf, member)))

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(crc, files["member"]))
    finally:
        for zf in opened:
            zf.close()


def other_snrs_check(
    index: pd.DataFrame, downloads: Path, workers: int, snrs: tuple[int, ...] = (0, 6)
) -> pd.DataFrame:
    """Counts the MIMII files at other SNRs whose rebuilt WAV matches a DCASE file.

    Uses the DCASE CRC-32 values of the committed mapping.

    Args:
        index: MIMII file index.
        downloads: Directory of the MIMII archives.
        workers: Reading threads.
        snrs: SNRs to check.

    Returns:
        One row per (machine type, SNR): MIMII files, distinct CRC-32 values,
        files whose CRC-32 is one of the DCASE files, and those files.
    """
    mapping = pd.read_csv(MAPPING_PATH, dtype=str)
    rows = []
    for machine_type in MACHINE_TYPES:
        dcase = set(
            mapping.loc[
                (mapping["machine_type"] == machine_type)
                & mapping["dcase_member"].notna(),
                "crc32",
            ].map(lambda v: int(v, 16))
        )
        for snr in snrs:
            files = index[
                (index["machine_type"] == machine_type) & (index["snr_db"] == snr)
            ].reset_index(drop=True)
            (archive,) = files["archive"].unique()
            crcs = mimii_crcs(files, downloads / archive, workers)
            hits = [m for m, c in zip(files["member"], crcs, strict=True) if c in dcase]
            rows.append(
                {
                    "machine_type": machine_type,
                    "snr_db": snr,
                    "n_mimii_files": len(files),
                    "n_distinct_crc32": len(set(crcs)),
                    "n_dcase_files": len(dcase),
                    "n_matching_dcase": len(hits),
                    "matching_members": ";".join(hits),
                }
            )
            logger.info("%s %+d dB: %s", machine_type, snr, rows[-1])
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    """Builds the DCASE 2020 -> MIMII -6 dB mapping.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        0 if every DCASE file matches exactly one MIMII -6 dB file, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description="Map DCASE 2020 files to MIMII.")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--other-snrs",
        action="store_true",
        help="Check that no MIMII 0 or +6 dB file matches a DCASE file.",
    )
    args = parser.parse_args(argv)
    paths = load_paths()
    setup_logging(paths.data_root / LOG_SUBDIR / "dcase2020.log")
    index = load_index(index_path())
    downloads = paths.data_root / DOWNLOAD_SUBDIR
    if args.other_snrs:
        check = other_snrs_check(index, downloads, args.workers)
        check.to_csv(OTHER_SNRS_PATH, index=False, lineterminator="\n")
        return 0 if (check["n_matching_dcase"] == 0).all() else 1
    tables = []
    for machine_type in MACHINE_TYPES:
        dcase = dcase_listing(machine_type)
        mimii = index[(index["machine_type"] == machine_type) & (index["snr_db"] == -6)]
        mimii = mimii.reset_index(drop=True)
        mimii["crc32"] = mimii_crcs(
            mimii, downloads / f"-6_dB_{machine_type}.zip", args.workers
        )
        if mimii["crc32"].duplicated().any():
            raise ValueError(f"{machine_type}: duplicated CRC-32 among MIMII files")
        merged = dcase.merge(
            mimii[["machine_id", "label", "clip", "crc32"]].rename(
                columns={
                    "machine_id": "mimii_machine_id",
                    "label": "mimii_label",
                    "clip": "mimii_clip",
                }
            ),
            on="crc32",
            how="outer",
            indicator=True,
        )
        merged["machine_type"] = machine_type
        logger.info("%s: %s", machine_type, merged["_merge"].value_counts().to_dict())
        tables.append(merged)
    mapping = pd.concat(tables, ignore_index=True)
    matched = mapping["_merge"] == "both"
    mapping["crc32"] = mapping["crc32"].map(lambda v: f"{int(v):08x}")
    mapping = mapping.drop(columns="_merge")
    mapping.to_csv(MAPPING_PATH, index=False, lineterminator="\n")
    consistent = (
        mapping.loc[matched, "machine_id"] == mapping.loc[matched, "mimii_machine_id"]
    ) & (
        mapping.loc[matched, "dcase_label"].map(
            {"normal": "normal", "anomaly": "abnormal"}
        )
        == mapping.loc[matched, "mimii_label"]
    )
    logger.info(
        "%d of %d rows matched; ID and label consistent for %d",
        int(matched.sum()),
        len(mapping),
        int(consistent.sum()),
    )
    return 0 if matched.all() and consistent.all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
