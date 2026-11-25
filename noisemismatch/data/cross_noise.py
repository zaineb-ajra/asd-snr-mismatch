"""Noise shared between test files and training files of other machine types.

Two of the five models (Wilkinghoff, OS-SCL) are trained on all machine types. A
factory-noise segment shared by a test file and a training file of another
machine type would be a shortcut available to them only. This module measures,
on the saved splits, for each SNR, the share of normal and of anomalous test files
that have a noise partner (same-SNR score >= threshold, the score of
:mod:`noisemismatch.data.matching`) among the training files of other machine
types, and of all machine types. The most negative score is also recorded as a
control: shared noise only gives positive correlations.

Usage::

    CUDA_VISIBLE_DEVICES=0 python -m noisemismatch.data.cross_noise

writes ``metadata/mimii_noise_check_cross_type.csv`` and exits with code 1 if the
normal/anomalous difference reaches the protocol limit.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from noisemismatch.config import CONFIG_DIR, load_paths, load_yaml
from noisemismatch.data.build_split import split_file_tables
from noisemismatch.data.download import DOWNLOAD_SUBDIR, LOG_SUBDIR
from noisemismatch.data.identity import load_identity
from noisemismatch.data.matching import block_scores, load_standardized
from noisemismatch.data.prepare import METADATA_DIR
from noisemismatch.data.split import FILE_COLUMNS, TEST, TRAIN
from noisemismatch.utils.logs import setup_logging

logger = logging.getLogger(__name__)

CROSS_CHECK_PATH: Path = METADATA_DIR / "mimii_noise_check_cross_type.csv"
FILE_KEY: list[str] = ["machine_type", *FILE_COLUMNS]
CHANNELS: int = 8
FRAMES: int = 160_000


def archive_members(files: pd.DataFrame, snr: int) -> pd.DataFrame:
    """Adds archive, member and audio-shape columns for :func:`load_standardized`."""
    out = files.copy()
    out["archive"] = [f"{snr}_dB_{t}.zip" for t in out["machine_type"]]
    out["member"] = [
        f"{t}/{i}/{lab}/{c}.wav"
        for t, i, lab, c in out[FILE_KEY].itertuples(index=False)
    ]
    out["channels"] = CHANNELS
    out["frames"] = FRAMES
    return out


def partner_statistics(
    scores: np.ndarray, test_types: np.ndarray, train_types: np.ndarray
) -> pd.DataFrame:
    """Best and most negative scores of each test file against training files.

    Args:
        scores: Scores shaped ``(n_test, n_train)``.
        test_types: Machine type of each test file.
        train_types: Machine type of each training file.

    Returns:
        Columns ``max_other_types``, ``min_other_types``, ``max_all_types`` and
        ``min_all_types`` (one row per test file).
    """
    other = test_types[:, None] != train_types[None, :]
    masked_max = np.where(other, scores, -np.inf)
    masked_min = np.where(other, scores, np.inf)
    return pd.DataFrame(
        {
            "max_other_types": masked_max.max(axis=1),
            "min_other_types": masked_min.min(axis=1),
            "max_all_types": scores.max(axis=1),
            "min_all_types": scores.min(axis=1),
        }
    )


def summarize(details: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Share of test files with a noise partner, per split, SNR, scope and label.

    Args:
        details: Per-test-file statistics with ``split_name``, ``snr_db`` and
            ``label`` columns.
        threshold: Score threshold of a noise partner.

    Returns:
        One row per (split, SNR, scope, label): ``n_test``, ``n_sharing``,
        ``share`` and, as a control, ``n_negative`` (a score <= -threshold).
    """
    rows = []
    for scope in ("other_types", "all_types"):
        grouped = details.groupby(["split_name", "snr_db", "label"])
        for (name, snr, label), part in grouped:
            sharing = int((part[f"max_{scope}"] >= threshold).sum())
            rows.append(
                {
                    "split_name": name,
                    "snr_db": snr,
                    "scope": scope,
                    "label": label,
                    "n_test": len(part),
                    "n_sharing": sharing,
                    "share": sharing / len(part),
                    "n_negative": int((part[f"min_{scope}"] <= -threshold).sum()),
                }
            )
    return pd.DataFrame(rows)


def largest_difference(summary: pd.DataFrame) -> float:
    """Largest |share(normal) - share(anomalous)| over splits, SNRs and scopes."""
    wide = summary.pivot_table(
        index=["split_name", "snr_db", "scope"], columns="label", values="share"
    )
    return float((wide["normal"] - wide["abnormal"]).abs().max())


def main(argv: list[str] | None = None) -> int:
    """Measures cross-type noise sharing on the saved splits.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        0 if the difference is below the protocol limit, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description="Cross-type noise reuse check.")
    parser.add_argument("--workers", type=int, default=12, help="Loading threads.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    args = parser.parse_args(argv)
    paths = load_paths()
    setup_logging(paths.data_root / LOG_SUBDIR / "cross_noise.log")
    torch.backends.cuda.matmul.allow_tf32 = False
    protocol = load_yaml(CONFIG_DIR / "protocol.yaml")
    threshold = float(protocol["split"]["noise_check"]["min_score"])
    limit = float(protocol["split"]["noise_check"]["max_difference"])
    tables = split_file_tables(load_identity(), protocol)
    downloads = paths.data_root / DOWNLOAD_SUBDIR

    details = []
    for snr in (0, 6, -6):
        parts = {n: t[t["snr_db"] == snr] for n, t in tables.items()}
        parts = {n: t for n, t in parts.items() if len(t)}
        union = (
            pd.concat(list(parts.values()))
            .drop_duplicates(FILE_KEY)
            .sort_values(FILE_KEY, ignore_index=True)
        )
        audio = load_standardized(archive_members(union, snr), downloads, args.workers)
        flat = audio.reshape(len(audio), -1)
        row = {tuple(k): n for n, k in enumerate(union[FILE_KEY].to_numpy())}
        logger.info("%+d dB: %d files loaded", snr, len(union))
        for name, part in parts.items():
            test = part[part["split"] == TEST]
            train = part[part["split"] == TRAIN]
            rows_test = np.array([row[tuple(k)] for k in test[FILE_KEY].to_numpy()])
            rows_train = np.array([row[tuple(k)] for k in train[FILE_KEY].to_numpy()])
            scores = block_scores(
                flat,
                flat,
                torch.device(args.device),
                rows_a=rows_test,
                rows_b=rows_train,
            )
            stats = partner_statistics(
                scores,
                test["machine_type"].to_numpy(),
                train["machine_type"].to_numpy(),
            )
            stats.insert(0, "split_name", name)
            stats.insert(1, "snr_db", snr)
            stats.insert(2, "label", test["label"].to_numpy())
            details.append(stats)
        del audio, flat

    summary = summarize(pd.concat(details, ignore_index=True), threshold)
    summary.to_csv(
        CROSS_CHECK_PATH, index=False, lineterminator="\n", float_format="%.5f"
    )
    difference = largest_difference(summary)
    logger.info("Cross-type noise check:\n%s", summary.round(4).to_string(index=False))
    logger.info(
        "Largest normal/anomalous difference: %.4f (limit %.2f)", difference, limit
    )
    return 0 if difference < limit else 1


if __name__ == "__main__":
    raise SystemExit(main())
