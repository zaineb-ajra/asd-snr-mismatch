"""Content-based comparison of MIMII files across SNRs: score matrices.

File names do not identify recordings across SNRs (docs/data.md), so clip
identity is established by content (docs/protocol.md, "Clip identity"). This
module computes the scores the identity rule needs; :mod:`noisemismatch.data.identity`
applies the rule.

The score of two files is the Pearson correlation of their 10-s waveforms at
lag 0, averaged over the 8 channels. Each file is standardized per channel (zero
mean, unit variance over time) and stored as float16; the score is then the dot
product of two standardized files divided by (channels x frames), computed in
float32 on the GPU.

For each machine type, all files (every machine ID, both labels) are compared,
so correspondences across IDs or labels are visible:

- ``s_p6``: 0 dB (rows) vs +6 dB (columns); ``s_0m``: 0 dB vs -6 dB;
  ``s_6m``: +6 dB vs -6 dB;
- ``s_00``, ``s_66``, ``s_mm``: within one SNR (duplicate recordings).

A channel-0 screen across machine types checks that no file matches a recording
of another machine type.

Usage::

    CUDA_VISIBLE_DEVICES=1 python -m noisemismatch.data.matching [--workers 12]

Outputs, under ``<data_root>/matching/``: ``<type>_files_<snr>.csv`` (row and
column order of the matrices), ``<type>_scores.npz`` and ``cross_type_ch0.csv``.
Peak memory is about 3 x 14 GB for fan (the largest type) plus 17 GB of
channel-0 copies kept for the cross-type screen.
"""

from __future__ import annotations

import argparse
import logging
import resource
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from noisemismatch.config import load_paths
from noisemismatch.data.download import DOWNLOAD_SUBDIR, LOG_SUBDIR
from noisemismatch.data.prepare import index_path, load_index
from noisemismatch.utils.logs import setup_logging

logger = logging.getLogger(__name__)

MATCHING_SUBDIR: str = "matching"
SNRS: tuple[int, int, int] = (0, 6, -6)
FILE_ORDER: list[str] = ["machine_id", "label", "clip"]
MATRICES: dict[str, tuple[int, int]] = {
    "s_p6": (0, 6),
    "s_0m": (0, -6),
    "s_6m": (6, -6),
    "s_00": (0, 0),
    "s_66": (6, 6),
    "s_mm": (-6, -6),
}


def standardize(samples: np.ndarray) -> np.ndarray:
    """Standardizes each channel of a clip over time.

    Args:
        samples: Array shaped ``(frames, channels)``.

    Returns:
        Float32 array shaped ``(channels, frames)`` with zero mean and unit
        variance per channel; a constant channel becomes all zeros.
    """
    x = samples.T.astype(np.float32)
    x -= x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return np.divide(x, std, out=np.zeros_like(x), where=std > 0)


def load_standardized(
    files: pd.DataFrame, downloads_dir: Path, workers: int
) -> np.ndarray:
    """Loads and standardizes files in parallel (threads).

    Args:
        files: Rows with ``archive``, ``member``, ``channels`` and ``frames``.
        downloads_dir: Directory holding the archives.
        workers: Number of threads (zlib and libsndfile release the GIL).

    Returns:
        Float16 array shaped ``(n_files, channels, frames)`` in row order.
    """
    channels = int(files["channels"].iloc[0])
    frames = int(files["frames"].iloc[0])
    out = np.empty((len(files), channels, frames), dtype=np.float16)
    local = threading.local()
    opened: list[zipfile.ZipFile] = []
    lock = threading.Lock()

    def handle(archive: str) -> zipfile.ZipFile:
        cache = getattr(local, "zips", None)
        if cache is None:
            cache = local.zips = {}
        if archive not in cache:
            cache[archive] = zipfile.ZipFile(downloads_dir / archive)
            with lock:
                opened.append(cache[archive])
        return cache[archive]

    def load(k: int, archive: str, member: str) -> None:
        with handle(archive).open(member) as f:
            samples, _ = sf.read(f, dtype="float32", always_2d=True)
        out[k] = standardize(samples)

    rows = zip(range(len(files)), files["archive"], files["member"], strict=True)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda row: load(*row), rows))
    finally:
        for zf in opened:
            zf.close()
    return out


def block_scores(
    a: np.ndarray,
    b: np.ndarray,
    device: torch.device,
    block: int = 512,
    rows_a: np.ndarray | None = None,
    rows_b: np.ndarray | None = None,
) -> np.ndarray:
    """Computes ``a @ b.T / a.shape[1]`` in float32, block by block.

    Args:
        a: Float16 array shaped ``(n_a, dim)`` (standardized, flattened files).
        b: Float16 array shaped ``(n_b, dim)``.
        device: Device for the products.
        block: Number of rows moved to the device at a time.
        rows_a: Optional rows of ``a`` to use (avoids copying a subset).
        rows_b: Optional rows of ``b`` to use.

    Returns:
        Float32 array shaped ``(len(rows_a), len(rows_b))`` (all rows by
        default): mean-channel correlations.
    """
    rows_a = np.arange(a.shape[0]) if rows_a is None else np.asarray(rows_a)
    rows_b = np.arange(b.shape[0]) if rows_b is None else np.asarray(rows_b)
    dim = a.shape[1]
    out = np.empty((len(rows_a), len(rows_b)), dtype=np.float32)
    with torch.no_grad():
        for j0 in range(0, len(rows_b), block):
            b_block = torch.from_numpy(b[rows_b[j0 : j0 + block]]).to(device).float()
            for i0 in range(0, len(rows_a), block):
                a_block = (
                    torch.from_numpy(a[rows_a[i0 : i0 + block]]).to(device).float()
                )
                product = (a_block @ b_block.T) / dim
                out[i0 : i0 + block, j0 : j0 + block] = product.cpu().numpy()
    return out


def type_files(index: pd.DataFrame, machine_type: str, snr_db: int) -> pd.DataFrame:
    """Files of one machine type at one SNR, in matrix order.

    Args:
        index: File index.
        machine_type: Machine type.
        snr_db: SNR.

    Returns:
        The files sorted by machine ID, label and clip name.
    """
    mask = (index["machine_type"] == machine_type) & (index["snr_db"] == snr_db)
    return index[mask].sort_values(FILE_ORDER, ignore_index=True)


def _max_rss_gb() -> float:
    """Peak resident memory of this process in GB (Linux reports kB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def compute_type(
    index: pd.DataFrame,
    downloads_dir: Path,
    machine_type: str,
    out_dir: Path,
    device: torch.device,
    workers: int,
) -> dict[int, np.ndarray]:
    """Computes and saves the score matrices of one machine type.

    Args:
        index: File index.
        downloads_dir: Directory holding the archives.
        machine_type: Machine type.
        out_dir: Output directory.
        device: Device for the products.
        workers: Loading threads.

    Returns:
        Channel 0 of the standardized files per SNR (float16, ``(n, frames)``),
        for the cross-type screen.
    """
    data: dict[int, np.ndarray] = {}
    for snr in SNRS:
        files = type_files(index, machine_type, snr)
        files[[*FILE_ORDER, "archive", "member"]].to_csv(
            out_dir / f"{machine_type}_files_{snr}.csv",
            index=False,
            lineterminator="\n",
        )
        data[snr] = load_standardized(files, downloads_dir, workers)
        logger.info(
            "%s %+d dB: %d files loaded (peak RSS %.1f GB)",
            machine_type,
            snr,
            len(files),
            _max_rss_gb(),
        )
    flat = {snr: z.reshape(len(z), -1) for snr, z in data.items()}
    scores = {
        name: block_scores(flat[r], flat[c], device)
        for name, (r, c) in MATRICES.items()
    }
    np.savez(out_dir / f"{machine_type}_scores.npz", **scores)
    logger.info("%s: score matrices saved", machine_type)
    return {snr: np.ascontiguousarray(z[:, 0, :]) for snr, z in data.items()}


def cross_type_screen(
    ch0: dict[str, dict[int, np.ndarray]],
    index: pd.DataFrame,
    device: torch.device,
) -> pd.DataFrame:
    """Compares every +6 and -6 dB file with the 0 dB files of all machine types.

    Args:
        ch0: Per machine type and SNR, standardized channel 0 of the files.
        index: File index (for file names).
        device: Device for the products.

    Returns:
        One row per compared file with the best score among 0 dB files of its
        own machine type and among those of the other types.
    """
    types = sorted(ch0)
    refs = np.concatenate([ch0[t][0] for t in types])
    ref_types = np.concatenate([[t] * len(ch0[t][0]) for t in types])
    rows = []
    for snr in (6, -6):
        files = np.concatenate([ch0[t][snr] for t in types])
        scores = block_scores(refs, files, device)
        names = pd.concat([type_files(index, t, snr) for t in types], ignore_index=True)
        for t in types:
            own = ref_types == t
            cols = np.flatnonzero(names["machine_type"].to_numpy() == t)
            best_own = scores[own][:, cols].max(axis=0)
            other = scores[~own][:, cols]
            best_other = other.max(axis=0)
            other_type = ref_types[~own][other.argmax(axis=0)]
            part = names.loc[cols, ["machine_type", *FILE_ORDER]].copy()
            part.insert(0, "snr_db", snr)
            part["best_own_type"] = best_own
            part["best_other_type"] = best_other
            part["other_type"] = other_type
            rows.append(part)
    return pd.concat(rows, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    """Computes all score matrices and the cross-type screen.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code (0).
    """
    parser = argparse.ArgumentParser(description="Score MIMII files across SNRs.")
    parser.add_argument("--workers", type=int, default=12, help="Loading threads.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    args = parser.parse_args(argv)

    paths = load_paths()
    setup_logging(paths.data_root / LOG_SUBDIR / "matching.log")
    torch.backends.cuda.matmul.allow_tf32 = False  # full float32 products
    device = torch.device(args.device)
    out_dir = paths.data_root / MATCHING_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    index = load_index(index_path())
    downloads_dir = paths.data_root / DOWNLOAD_SUBDIR

    sizes = index.groupby("machine_type").size().sort_values(ascending=False)
    ch0: dict[str, dict[int, np.ndarray]] = {}
    for machine_type in sizes.index:  # largest type first: lowest memory peak
        ch0[machine_type] = compute_type(
            index, downloads_dir, machine_type, out_dir, device, args.workers
        )
    screen = cross_type_screen(ch0, index, device)
    screen.to_csv(
        out_dir / "cross_type_ch0.csv",
        index=False,
        lineterminator="\n",
        float_format="%.5f",
    )
    logger.info("Done (peak RSS %.1f GB)", _max_rss_gb())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
