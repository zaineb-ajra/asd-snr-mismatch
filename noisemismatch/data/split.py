"""Train/test split defined on clip identity, with leakage checks.

For each (machine type, machine ID), every anomalous clip goes to the test set,
together with the same number of normal clips drawn uniformly without replacement
with a fixed seed; the remaining normal clips form the training set (MIMII paper,
Sec. 4). The split is defined on clip *identity* (machine type, machine ID, label,
clip name), not on files: MIMII provides every clip at three SNRs, and splitting
files would let a test clip leak into training at another SNR.

Tables used here are pandas DataFrames:

- *identity table*: one row per clip identity, columns ``IDENTITY_COLUMNS``;
- *split table*: identity table plus a ``split`` column (``train`` / ``test``);
- *file table*: one row per audio file, i.e. identity columns plus ``snr_db``
  (and any location columns).
"""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd

IDENTITY_COLUMNS: tuple[str, ...] = ("machine_type", "machine_id", "label", "clip")
GROUP_COLUMNS: tuple[str, ...] = ("machine_type", "machine_id")
NORMAL: str = "normal"
ANOMALOUS: str = "abnormal"
TRAIN: str = "train"
TEST: str = "test"


def _require_columns(table: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Raises ValueError if ``table`` lacks any of ``columns``."""
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")


def _group_seed(seed: int, machine_type: str, machine_id: str) -> list[int]:
    """Derives a per-group seed that does not depend on the other groups."""
    key = zlib.crc32(f"{machine_type}/{machine_id}".encode())
    return [seed, key]


def make_split(identities: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Assigns every clip identity to the training or the test set.

    Each (machine type, machine ID) group uses its own random generator, seeded
    from ``seed`` and the group name, so a group's draw depends neither on the
    row order of ``identities`` nor on the other groups.

    Args:
        identities: Identity table (one row per clip identity).
        seed: Seed of the draw of normal test clips.

    Returns:
        The split table, sorted by identity columns.

    Raises:
        ValueError: On duplicated identities, unknown labels, or a group with
            more anomalous than normal clips.
    """
    _require_columns(identities, IDENTITY_COLUMNS)
    table = identities.loc[:, list(IDENTITY_COLUMNS)].copy()
    if table.duplicated().any():
        raise ValueError("duplicated clip identities")
    unknown = set(table["label"]) - {NORMAL, ANOMALOUS}
    if unknown:
        raise ValueError(f"unknown labels: {sorted(unknown)}")
    table = table.sort_values(list(IDENTITY_COLUMNS), ignore_index=True)
    table["split"] = TRAIN
    table.loc[table["label"] == ANOMALOUS, "split"] = TEST

    for (machine_type, machine_id), group in table.groupby(
        list(GROUP_COLUMNS), sort=True
    ):
        normal_index = group.index[group["label"] == NORMAL]
        n_test = int((group["label"] == ANOMALOUS).sum())
        if n_test > len(normal_index):
            raise ValueError(
                f"{machine_type}/{machine_id}: {n_test} anomalous clips but only "
                f"{len(normal_index)} normal clips"
            )
        rng = np.random.default_rng(_group_seed(seed, machine_type, machine_id))
        chosen = rng.choice(normal_index.to_numpy(), size=n_test, replace=False)
        table.loc[chosen, "split"] = TEST
    return table


def check_split(split: pd.DataFrame, require_balance: bool = True) -> None:
    """Checks the invariants of a split table.

    Args:
        split: Split table. If it has a ``cluster`` column, clusters must not be
            split across training and test.
        require_balance: Require as many normal as anomalous test clips per
            group (cluster splits may deviate; see :func:`make_cluster_split`).

    Raises:
        ValueError: If identities are duplicated, an anomalous clip is in the
            training set, a cluster is split, the numbers of normal and anomalous
            test clips differ in a group (when required), or a group has no
            training clip.
    """
    _require_columns(split, (*IDENTITY_COLUMNS, "split"))
    if split.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("duplicated clip identities")
    if not set(split["split"]) <= {TRAIN, TEST}:
        raise ValueError(f"unknown split values: {sorted(set(split['split']))}")
    if ((split["label"] == ANOMALOUS) & (split["split"] == TRAIN)).any():
        raise ValueError("anomalous clip in the training set")
    for key, group in split.groupby(list(GROUP_COLUMNS), sort=True):
        test = group[group["split"] == TEST]
        n_normal = int((test["label"] == NORMAL).sum())
        n_anomalous = int((test["label"] == ANOMALOUS).sum())
        if require_balance and n_normal != n_anomalous:
            raise ValueError(
                f"{key}: {n_normal} normal vs {n_anomalous} anomalous test clips"
            )
        if not (group["split"] == TRAIN).any():
            raise ValueError(f"{key}: no training clip")
    if "cluster" in split.columns:
        sides = split.groupby("cluster")["split"].nunique()
        if (sides > 1).any():
            raise ValueError(f"{int((sides > 1).sum())} cluster(s) split across sets")


def assign_files(split: pd.DataFrame, files: pd.DataFrame) -> pd.DataFrame:
    """Propagates the split from clip identities to audio files.

    Args:
        split: Split table.
        files: File table (identity columns, ``snr_db`` and location columns).

    Returns:
        ``files`` with a ``split`` column, in the same row order.

    Raises:
        ValueError: If a file has no entry in the split, or a split identity
            is not present at every SNR found in ``files``.
    """
    _require_columns(files, (*IDENTITY_COLUMNS, "snr_db"))
    keys = list(IDENTITY_COLUMNS)
    merged = files.merge(
        split.loc[:, [*keys, "split"]], on=keys, how="left", validate="many_to_one"
    )
    if merged["split"].isna().any():
        n = int(merged["split"].isna().sum())
        raise ValueError(f"{n} file(s) have no entry in the split")
    snrs = sorted(files["snr_db"].unique())
    present = files.groupby(keys)["snr_db"].agg(lambda s: sorted(s.unique()))
    per_identity = split.set_index(keys).index
    incomplete = [k for k in per_identity if present.get(k, []) != snrs]
    if incomplete:
        raise ValueError(
            f"{len(incomplete)} clip(s) not present at every SNR {snrs}, "
            f"e.g. {incomplete[:3]}"
        )
    return merged


def check_no_leakage(files: pd.DataFrame) -> None:
    """Checks that no clip identity is used for training at any SNR and tested.

    Args:
        files: File table with a ``split`` column.

    Raises:
        ValueError: If an identity is in the training set at one SNR and in the
            test set at the same or another SNR.
    """
    _require_columns(files, (*IDENTITY_COLUMNS, "snr_db", "split"))
    keys = list(IDENTITY_COLUMNS)
    train = set(files.loc[files["split"] == TRAIN, keys].itertuples(index=False))
    test = set(files.loc[files["split"] == TEST, keys].itertuples(index=False))
    leaked = train & test
    if leaked:
        raise ValueError(
            f"{len(leaked)} clip(s) in both training and test sets, "
            f"e.g. {sorted(leaked)[:3]}"
        )


def save_split(split: pd.DataFrame, path: Path) -> None:
    """Writes a split table as CSV, sorted, with Unix line endings.

    Args:
        split: Split table.
        path: Output CSV file.
    """
    columns = [*IDENTITY_COLUMNS, *(["cluster"] if "cluster" in split else []), "split"]
    table = split.loc[:, columns].sort_values(list(IDENTITY_COLUMNS))
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, lineterminator="\n")


def load_split(path: Path, require_balance: bool = True) -> pd.DataFrame:
    """Reads a split table written by ``save_split`` and checks its invariants.

    Args:
        path: CSV file.
        require_balance: Forwarded to :func:`check_split`.

    Returns:
        The split table (all columns as strings).
    """
    split = pd.read_csv(path, dtype=str, keep_default_na=False)
    check_split(split, require_balance=require_balance)
    return split


FILE_COLUMNS: tuple[str, ...] = ("file_machine_id", "file_label", "file_clip")


def identity_files(
    identities: pd.DataFrame,
    snrs: tuple[int, ...] = (0, 6, -6),
    only_kept: bool = True,
) -> pd.DataFrame:
    """Lists, for each kept identity, the file that holds it at each SNR.

    Identities come from the content-based identity table
    (``metadata/mimii_identity.csv``): the reference file at 0 dB, the accepted
    matches ``p6_*`` at +6 dB and ``m6_*`` at -6 dB.

    Args:
        identities: Identity table with columns ``machine_type``,
            ``machine_id``, ``label``, ``clip``, ``kept`` and the ``p6_`` / ``m6_``
            machine ID, label and clip columns.
        snrs: SNRs to list (0 = reference, 6 = ``p6_*``, -6 = ``m6_*``).
        only_kept: Keep only rows whose ``kept`` column is true; otherwise use
            every given row.

    Returns:
        One row per (kept identity, SNR): identity columns, ``snr_db`` and the
        file actually used (``file_machine_id``, ``file_label``, ``file_clip``).
        The label of a clip is always the label of its identity.
    """
    kept = (
        identities[identities["kept"].astype(str) == "True"]
        if only_kept
        else identities
    )
    prefixes = {0: None, 6: "p6", -6: "m6"}
    parts = []
    for snr in snrs:
        prefix = prefixes[snr]
        part = kept.loc[:, list(IDENTITY_COLUMNS)].copy()
        part["snr_db"] = snr
        for column, key in zip(
            FILE_COLUMNS, ("machine_id", "label", "clip"), strict=True
        ):
            source = key if prefix is None else f"{prefix}_{key}"
            part[column] = kept[source].to_numpy()
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def check_content_identity(files: pd.DataFrame) -> None:
    """Checks that identities and files correspond one to one at every SNR.

    Args:
        files: Output of :func:`identity_files` (optionally with ``split``).

    Raises:
        ValueError: If an identity has zero or several files at some SNR, or a
            file is used by several identities.
    """
    _require_columns(files, (*IDENTITY_COLUMNS, "snr_db", *FILE_COLUMNS))
    snrs = sorted(files["snr_db"].unique())
    per_identity = files.groupby(list(IDENTITY_COLUMNS))["snr_db"].agg(list)
    bad = per_identity[per_identity.map(lambda v: sorted(v) != snrs)]
    if not bad.empty:
        raise ValueError(f"{len(bad)} identities without exactly one file per SNR")
    file_keys = ["machine_type", "snr_db", *FILE_COLUMNS]
    shared = files.duplicated(file_keys, keep=False)
    if shared.any():
        raise ValueError(f"{int(shared.sum())} rows use a file shared by identities")


def make_cluster_split(identities: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Splits clusters of identities into training and test sets.

    Per (machine type, machine ID): clusters containing an anomalous clip go to
    the test set; normal-only clusters are then visited in an order drawn with a
    per-group generator (seeded from ``seed`` and the group name) and added to
    the test set if they do not exceed the remaining need (number of anomalous
    minus normal test clips). If the need is not met exactly, the smallest
    remaining cluster is added when that brings the count closer.

    Args:
        identities: Identity columns plus ``cluster`` (clusters must not span
            machine IDs).
        seed: Seed of the draw.

    Returns:
        Split table with ``cluster`` and ``split`` columns, sorted by identity.

    Raises:
        ValueError: On duplicated identities, unknown labels, or a cluster
            spanning several machine IDs.
    """
    _require_columns(identities, (*IDENTITY_COLUMNS, "cluster"))
    table = identities.loc[:, [*IDENTITY_COLUMNS, "cluster"]].copy()
    if table.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("duplicated clip identities")
    unknown = set(table["label"]) - {NORMAL, ANOMALOUS}
    if unknown:
        raise ValueError(f"unknown labels: {sorted(unknown)}")
    spans = table.groupby("cluster")[list(GROUP_COLUMNS)].nunique().max(axis=1)
    if (spans > 1).any():
        raise ValueError("a cluster spans several machine IDs")
    table = table.sort_values(list(IDENTITY_COLUMNS), ignore_index=True)
    table["split"] = TRAIN

    for (machine_type, machine_id), group in table.groupby(
        list(GROUP_COLUMNS), sort=True
    ):
        per_cluster = group.groupby("cluster", sort=True)["label"]
        n_anomalous = per_cluster.apply(lambda s: int((s == ANOMALOUS).sum()))
        n_normal = per_cluster.apply(lambda s: int((s == NORMAL).sum()))
        forced = n_anomalous.index[n_anomalous > 0]
        need = int(n_anomalous.sum() - n_normal[forced].sum())
        candidates = n_normal.index[n_anomalous == 0].to_numpy()
        rng = np.random.default_rng(_group_seed(seed, machine_type, machine_id))
        chosen = list(forced)
        remaining = []
        for cluster in rng.permutation(candidates):
            size = int(n_normal[cluster])
            if need > 0 and size <= need:
                chosen.append(cluster)
                need -= size
            else:
                remaining.append(cluster)
        if need > 0 and remaining:
            smallest = min(remaining, key=lambda c: (int(n_normal[c]), c))
            if int(n_normal[smallest]) - need < need:
                chosen.append(smallest)
        in_test = group.index[group["cluster"].isin(chosen)]
        table.loc[in_test, "split"] = TEST
    return table


def split_balance(split: pd.DataFrame) -> pd.DataFrame:
    """Counts test and training clips per (machine type, machine ID).

    Args:
        split: Split table.

    Returns:
        Columns ``train_normal``, ``test_normal``, ``test_abnormal`` and
        ``test_difference`` (normal minus anomalous test clips).
    """
    counts = split.groupby([*GROUP_COLUMNS, "split", "label"]).size()
    table = counts.unstack(["split", "label"], fill_value=0)
    out = pd.DataFrame(
        {
            "train_normal": table.get((TRAIN, NORMAL), 0),
            "test_normal": table.get((TEST, NORMAL), 0),
            "test_abnormal": table.get((TEST, ANOMALOUS), 0),
        }
    )
    out["test_difference"] = out["test_normal"] - out["test_abnormal"]
    return out.reset_index()
