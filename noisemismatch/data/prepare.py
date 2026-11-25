"""Data preparation: file index, inventory and clip pairing across SNRs.

Usage::

    python -m noisemismatch.data.prepare [--reindex]

Outputs:

- ``<data_root>/index/files.csv``: one row per WAV file (regenerable, not
  versioned);
- ``metadata/mimii_inventory.csv``: file counts and audio properties per
  (machine type, machine ID, label, SNR);
- ``metadata/mimii_pairing_content.csv``: content check of the name-based pairing
  of clips across SNRs, on a random sample.

File names do not identify recordings across SNRs (docs/data.md), so the
train/test split is built on content-based clip identities, in a separate step.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from noisemismatch.config import CONFIG_DIR, REPO_ROOT, load_paths, load_yaml
from noisemismatch.data.download import DOWNLOAD_SUBDIR, LOG_SUBDIR, load_manifest
from noisemismatch.data.mimii import build_file_index
from noisemismatch.data.pairing import content_pairing, unpaired_identities
from noisemismatch.data.split import (
    ANOMALOUS,
    IDENTITY_COLUMNS,
    NORMAL,
    TEST,
    TRAIN,
)
from noisemismatch.utils.logs import setup_logging

logger = logging.getLogger(__name__)

METADATA_DIR: Path = REPO_ROOT / "metadata"
SPLIT_PATH: Path = METADATA_DIR / "mimii_split.csv"
INDEX_RELPATH: str = "index/files.csv"
AUDIO_COLUMNS: tuple[str, ...] = (
    "format_tag",
    "extensible",
    "channels",
    "sample_rate",
    "bits_per_sample",
    "frames",
    "file_bytes",
)
_STRING_COLUMNS = {c: str for c in (*IDENTITY_COLUMNS, "archive", "member")}


def index_path() -> Path:
    """Returns the location of the file index under the configured ``data_root``."""
    return load_paths().data_root / INDEX_RELPATH


def load_index(path: Path) -> pd.DataFrame:
    """Reads a file index saved as CSV, keeping identifiers as strings.

    Args:
        path: CSV file written by :func:`main`.

    Returns:
        The file index.
    """
    return pd.read_csv(path, dtype=_STRING_COLUMNS)


def inventory(index: pd.DataFrame) -> pd.DataFrame:
    """Counts files and summarizes audio properties per group.

    Args:
        index: File index.

    Returns:
        One row per (machine type, machine ID, label, SNR) with ``n_files`` and,
        for each audio property, its value (or ``"mixed"`` if it varies).
    """
    keys = ["machine_type", "machine_id", "label", "snr_db"]
    grouped = index.groupby(keys, sort=True)
    table = grouped.size().rename("n_files").to_frame()
    for column in AUDIO_COLUMNS:
        values = grouped[column].agg(
            lambda s: s.iloc[0] if s.nunique() == 1 else "mixed"
        )
        table[column] = values
    table["duration_s_min"] = grouped["duration_s"].min()
    table["duration_s_max"] = grouped["duration_s"].max()
    return table.reset_index()


def split_summary(files: pd.DataFrame) -> pd.DataFrame:
    """Counts clip identities per split and label for each machine type and ID.

    Args:
        files: File table with a ``split`` column.

    Returns:
        One row per (machine type, machine ID) with columns ``train_normal``,
        ``test_normal`` and ``test_abnormal`` (clip identities, i.e. per SNR).
    """
    clips = files.drop_duplicates(list(IDENTITY_COLUMNS))
    counts = clips.groupby(["machine_type", "machine_id", "split", "label"]).size()
    table = counts.unstack(["split", "label"], fill_value=0)
    return pd.DataFrame(
        {
            "train_normal": table.get((TRAIN, NORMAL), 0),
            "test_normal": table.get((TEST, NORMAL), 0),
            "test_abnormal": table.get((TEST, ANOMALOUS), 0),
        }
    ).reset_index()


def _check_downloads(downloads_dir: Path, filenames: set[str]) -> None:
    """Raises RuntimeError unless every archive passed the MD5 verification."""
    report_path = downloads_dir / "verification.json"
    if not report_path.exists():
        raise RuntimeError("no verification report: run noisemismatch.data.download")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    verified = {r["filename"] for r in report["results"] if r["ok"]}
    missing = sorted(filenames - verified)
    if missing:
        raise RuntimeError(f"archives not verified: {missing}")


def main(argv: list[str] | None = None) -> int:
    """Runs the data preparation.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 if every check passed.
    """
    parser = argparse.ArgumentParser(description="Index, inventory and split MIMII.")
    parser.add_argument(
        "--reindex", action="store_true", help="Rebuild the file index."
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Archives indexed in parallel."
    )
    parser.add_argument(
        "--pairing-per-group",
        type=int,
        default=5,
        help="Clips drawn per (machine type, ID, label) for the content check.",
    )
    args = parser.parse_args(argv)

    paths = load_paths()
    downloads_dir = paths.data_root / DOWNLOAD_SUBDIR
    setup_logging(paths.data_root / LOG_SUBDIR / "prepare.log")
    protocol = load_yaml(CONFIG_DIR / "protocol.yaml")
    pairing_seed = int(protocol["split"]["seed"])

    archives = load_manifest()
    _check_downloads(downloads_dir, {a.filename for a in archives})

    index_file = index_path()
    if index_file.exists() and not args.reindex:
        logger.info("Reading file index %s", index_file)
        index = load_index(index_file)
    else:
        logger.info("Indexing %d archives", len(archives))
        index = build_file_index(downloads_dir, archives, workers=args.workers)
        index_file.parent.mkdir(parents=True, exist_ok=True)
        index.to_csv(index_file, index=False, lineterminator="\n")
    logger.info("%d WAV files indexed", len(index))

    METADATA_DIR.mkdir(exist_ok=True)
    inventory(index).to_csv(
        METADATA_DIR / "mimii_inventory.csv", index=False, lineterminator="\n"
    )
    for column in AUDIO_COLUMNS:
        logger.info("%s: %s", column, sorted(index[column].unique().tolist()))

    unpaired = unpaired_identities(index)
    if not unpaired.empty:
        unpaired.to_csv(
            METADATA_DIR / "mimii_unpaired.csv", index=False, lineterminator="\n"
        )
        logger.error("%d clip(s) are not present at every SNR", len(unpaired))
        return 1
    logger.info("Every clip identity is present at every SNR")

    pairing = content_pairing(
        index, downloads_dir, args.pairing_per_group, seed=pairing_seed
    )
    pairing.to_csv(
        METADATA_DIR / "mimii_pairing_content.csv",
        index=False,
        lineterminator="\n",
        float_format="%.6f",
    )
    numeric = pairing.select_dtypes("number")
    logger.info("Content pairing check:\n%s", numeric.describe().to_string())

    # File names do not identify recordings across SNRs (docs/data.md), so the
    # split is built on content-based clip identities, in a separate step.
    logger.info("Split: built from content-based clip identities (separate step)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
