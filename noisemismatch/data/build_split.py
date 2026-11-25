"""Builds the train/test splits from content-based clip identities.

Implements "Train/test split" and "Sensitivity to the exclusion of clips" of
docs/protocol.md:

1. identity sets: the main set (identities kept at +6 and -6 dB) and the
   sensitivity set (identities matched at +6 dB), both without the excluded
   machine IDs;
2. clusters: identities whose accepted +6 dB files score >= the cluster threshold
   are linked; connected components are clusters;
3. candidate cluster splits (:func:`noisemismatch.data.split.make_cluster_split`);
4. noise reuse check on the candidates.

Usage::

    python -m noisemismatch.data.build_split          # candidates + checks only
    python -m noisemismatch.data.build_split --save   # then save (never overwrites)

Saved files: ``metadata/mimii_split.csv``, ``metadata/mimii_split_sensitivity.csv``,
``metadata/mimii_split_summary.csv`` and ``metadata/mimii_noise_check.csv``.
Requires the score matrices of :mod:`noisemismatch.data.matching` and the identity
table of :mod:`noisemismatch.data.identity`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from noisemismatch.config import CONFIG_DIR, load_paths, load_yaml
from noisemismatch.data.download import LOG_SUBDIR
from noisemismatch.data.identity import FILE_KEYS, file_positions, load_identity
from noisemismatch.data.matching import MATCHING_SUBDIR
from noisemismatch.data.prepare import METADATA_DIR, SPLIT_PATH
from noisemismatch.data.split import (
    FILE_COLUMNS,
    IDENTITY_COLUMNS,
    NORMAL,
    TRAIN,
    assign_files,
    check_content_identity,
    check_no_leakage,
    check_split,
    identity_files,
    load_split,
    make_cluster_split,
    make_split,
    save_split,
    split_balance,
)
from noisemismatch.utils.logs import setup_logging

logger = logging.getLogger(__name__)

SENSITIVITY_SPLIT_PATH: Path = METADATA_DIR / "mimii_split_sensitivity.csv"
SUMMARY_PATH: Path = METADATA_DIR / "mimii_split_summary.csv"
NOISE_CHECK_PATH: Path = METADATA_DIR / "mimii_noise_check.csv"
WITHIN_SNR_MATRIX: dict[int, str] = {0: "s_00", 6: "s_66", -6: "s_mm"}


def identity_sets(
    identity: pd.DataFrame, excluded: list[dict[str, str]]
) -> dict[str, pd.DataFrame]:
    """Selects the identities of the main and sensitivity splits.

    Args:
        identity: Identity table.
        excluded: Machine IDs removed from every analysis.

    Returns:
        ``{"main": kept identities, "sensitivity": identities matched at +6 dB}``,
        both without the excluded machine IDs.
    """
    drop = np.zeros(len(identity), dtype=bool)
    for item in excluded:
        drop |= (identity["machine_type"] == item["machine_type"]) & (
            identity["machine_id"] == item["machine_id"]
        )
    usable = identity[~drop]
    return {
        "main": usable[usable["kept"]].reset_index(drop=True),
        "sensitivity": usable[usable["p6_accepted"]].reset_index(drop=True),
    }


def add_clusters(
    identities: pd.DataFrame, matching_dir: Path, threshold: float
) -> pd.DataFrame:
    """Adds a ``cluster`` column linking identities whose +6 dB files score high.

    Args:
        identities: Identity rows with accepted +6 dB matches (``p6_*`` columns).
        matching_dir: Directory of the score matrices.
        threshold: Minimum +6 dB score linking two identities.

    Returns:
        A copy of ``identities`` with ``cluster`` = ``<machine type>:<number>``.
    """
    out = identities.copy()
    out["cluster"] = ""
    for machine_type, rows in out.groupby("machine_type"):
        position = file_positions(matching_dir, machine_type, 6)
        index = np.array(
            [
                position[tuple(r)]
                for r in rows[[f"p6_{k}" for k in FILE_KEYS]].to_numpy()
            ]
        )
        scores = np.load(matching_dir / f"{machine_type}_scores.npz")["s_66"]
        linked = scores[np.ix_(index, index)] >= threshold
        np.fill_diagonal(linked, False)
        _, component = connected_components(csr_matrix(linked), directed=False)
        out.loc[rows.index, "cluster"] = [f"{machine_type}:{c:05d}" for c in component]
    return out


def noise_check(
    files: pd.DataFrame, matching_dir: Path, min_score: float
) -> pd.DataFrame:
    """Share of test files sharing a noise segment with a training file.

    Two files of one SNR share a noise segment if their score at that SNR is at
    least ``min_score``; files sharing machine sound are in the same cluster,
    hence on the same side of the split.

    Args:
        files: Files of a split (:func:`identity_files` + ``split``).
        matching_dir: Directory of the score matrices.
        min_score: Score threshold.

    Returns:
        One row per (SNR, scope, label) with ``n_test``, ``n_sharing`` and
        ``share``; scope is ``machine_id`` or ``machine_type`` (training files of
        the same machine ID, or of any ID of the machine type).
    """
    records = []
    for (machine_type, snr), part in files.groupby(["machine_type", "snr_db"]):
        position = file_positions(matching_dir, machine_type, snr)
        rows = np.array(
            [position[tuple(r)] for r in part[list(FILE_COLUMNS)].to_numpy()]
        )
        matrix = np.load(matching_dir / f"{machine_type}_scores.npz")[
            WITHIN_SNR_MATRIX[snr]
        ]
        is_train = (part["split"] == TRAIN).to_numpy()
        ids = part["machine_id"].to_numpy()
        for k in np.flatnonzero(~is_train):
            scores = matrix[rows[k], rows[is_train]]
            same_id = ids[is_train] == ids[k]
            records.append(
                {
                    "snr_db": snr,
                    "machine_type": machine_type,
                    "machine_id": ids[k],
                    "label": part["label"].iloc[k],
                    "machine_id_scope": bool((scores[same_id] >= min_score).any()),
                    "machine_type_scope": bool((scores >= min_score).any()),
                }
            )
    table = pd.DataFrame(records)
    summary = []
    for scope in ("machine_id_scope", "machine_type_scope"):
        grouped = table.groupby(["snr_db", "label"])[scope].agg(["size", "sum"])
        for (snr, label), row in grouped.iterrows():
            summary.append(
                {
                    "snr_db": snr,
                    "scope": scope.removesuffix("_scope"),
                    "label": label,
                    "n_test": int(row["size"]),
                    "n_sharing": int(row["sum"]),
                    "share": row["sum"] / row["size"],
                }
            )
    return pd.DataFrame(summary)


def max_share_difference(check: pd.DataFrame) -> float:
    """Largest |share(normal) - share(anomalous)| over SNRs and scopes."""
    keys = [k for k in ("split_name", "snr_db", "scope") if k in check.columns]
    wide = check.pivot_table(index=keys, columns="label", values="share")
    return float((wide[NORMAL] - wide["abnormal"]).abs().max())


def build(
    matching_dir: Path, protocol: dict
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    """Builds and checks the candidate splits (nothing is written).

    Args:
        matching_dir: Directory of the score matrices.
        protocol: Parsed ``configs/protocol.yaml``.

    Returns:
        ``(splits, files, noise)``: split tables and file tables per split name,
        and the concatenated noise reuse check.
    """
    split_cfg = protocol["split"]
    sets = identity_sets(load_identity(), protocol["dataset"]["excluded_machine_ids"])
    snrs = {
        "main": (0, 6, -6),
        "sensitivity": tuple(protocol["sensitivity"]["snrs_db"]),
    }
    splits, files, checks = {}, {}, []
    for name, identities in sets.items():
        threshold = split_cfg.get("cluster_min_score_p6")
        if threshold is None:  # clusters removed (protocol amendment 1)
            split = make_split(
                identities[list(IDENTITY_COLUMNS)], int(split_cfg["seed"])
            )
            check_split(split)
        else:
            clustered = add_clusters(identities, matching_dir, float(threshold))
            split = make_cluster_split(
                clustered[[*IDENTITY_COLUMNS, "cluster"]], int(split_cfg["seed"])
            )
            check_split(split, require_balance=False)
        split_files = assign_files(
            split, identity_files(identities, snrs=snrs[name], only_kept=False)
        )
        check_content_identity(split_files)
        check_no_leakage(split_files)
        check = noise_check(
            split_files, matching_dir, float(split_cfg["noise_check"]["min_score"])
        )
        check.insert(0, "split_name", name)
        splits[name], files[name] = split, split_files
        checks.append(check)
        logger.info("%s: %d identities", name, len(split))
    return splits, files, pd.concat(checks, ignore_index=True)


def split_file_tables(
    identity: pd.DataFrame, protocol: dict
) -> dict[str, pd.DataFrame]:
    """Files of the saved main and sensitivity splits, with their partition.

    Args:
        identity: Identity table.
        protocol: Parsed ``configs/protocol.yaml``.

    Returns:
        ``{"main": files, "sensitivity": files}`` (identity columns, ``snr_db``,
        file columns and ``split``).
    """
    sets = identity_sets(identity, protocol["dataset"]["excluded_machine_ids"])
    snrs = {
        "main": (0, 6, -6),
        "sensitivity": tuple(protocol["sensitivity"]["snrs_db"]),
    }
    paths = {"main": SPLIT_PATH, "sensitivity": SENSITIVITY_SPLIT_PATH}
    return {
        name: assign_files(
            load_split(paths[name]),
            identity_files(sets[name], snrs=snrs[name], only_kept=False),
        )
        for name in paths
    }


def main(argv: list[str] | None = None) -> int:
    """Builds the candidate splits, runs the checks and optionally saves them.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        0 on success; 1 if the noise check stops the procedure or a split file
        already exists when saving.
    """
    parser = argparse.ArgumentParser(description="Build the MIMII splits.")
    parser.add_argument("--save", action="store_true", help="Save the splits.")
    args = parser.parse_args(argv)
    paths = load_paths()
    setup_logging(paths.data_root / LOG_SUBDIR / "build_split.log")
    protocol = load_yaml(CONFIG_DIR / "protocol.yaml")
    splits, _, noise = build(paths.data_root / MATCHING_SUBDIR, protocol)

    summary = pd.concat(
        [split_balance(s).assign(split_name=n) for n, s in splits.items()],
        ignore_index=True,
    )
    logger.info("Split sizes:\n%s", summary.to_string(index=False))
    logger.info("Noise reuse check:\n%s", noise.round(4).to_string(index=False))
    difference = max_share_difference(noise)
    limit = float(protocol["split"]["noise_check"]["max_difference"])
    logger.info(
        "Largest normal/anomalous difference: %.4f (limit %.2f)", difference, limit
    )
    if difference >= limit:
        logger.error("Noise reuse check failed: review before saving any split")
        return 1
    if not args.save:
        logger.info("Checks passed; run with --save to write the splits")
        return 0
    for path in (SPLIT_PATH, SENSITIVITY_SPLIT_PATH):
        if path.exists():
            logger.error("%s already exists; splits are never regenerated", path)
            return 1
    save_split(splits["main"], SPLIT_PATH)
    save_split(splits["sensitivity"], SENSITIVITY_SPLIT_PATH)
    summary.to_csv(SUMMARY_PATH, index=False, lineterminator="\n")
    noise.to_csv(
        NOISE_CHECK_PATH, index=False, lineterminator="\n", float_format="%.5f"
    )
    logger.info("Saved %s and %s", SPLIT_PATH, SENSITIVITY_SPLIT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
