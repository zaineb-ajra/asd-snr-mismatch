"""Analysis of the study runs (Phase 7), from the saved per-clip scores only.

Reads the ``scores.csv`` of the Phase 6 runs; no model is rerun. Definitions:
``docs/protocol.md`` ("Metrics", "Aggregation", "Analyses added after the
results"):

- per-ID AUC and pAUC, with the pre-registered 95 % bootstrap interval of the
  AUC (:func:`per_id`);
- aggregation: mean over the IDs of a machine type, then over machine types,
  per run and test SNR (:func:`aggregated`);
- ΔAUC computed seed by seed against the matched model at the test SNR, then
  mean and sample standard deviation over seeds (:func:`deltas`);
- secondary analysis decided after the stage-A results: ratio of the errors
  1 - AUC of the aggregated AUCs, seed by seed (:func:`error_ratios`);
- interval on aggregated values, decided after the results (2026-10-04): paired
  bootstrap of the test clips, the same draws for every model, condition, seed
  and test SNR (:func:`paired_bootstrap`).
"""

from __future__ import annotations

import multiprocessing
import zlib
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from noisemismatch.eval.aggregate import aggregate, drop_ids, seed_summary
from noisemismatch.eval.evaluate import per_id_table
from noisemismatch.eval.metrics import pauc

MODELS: tuple[str, ...] = ("ae", "mobilenetv2", "eat_knn", "wilkinghoff", "osscl")
MODEL_NAMES: dict[str, str] = {
    "ae": "Autoencoder",
    "mobilenetv2": "MobileNetV2",
    "eat_knn": "EAT + kNN",
    "wilkinghoff": "Wilkinghoff",
    "osscl": "OS-SCL",
}
# Training condition -> training SNR (None: multi-condition).
CONDITION_SNR: dict[str, int | None] = {"m6dB": -6, "0dB": 0, "p6dB": 6, "multi": None}
RUN_KEYS: list[str] = ["model", "train_condition", "seed"]
CELL_KEYS: list[str] = ["model", "train_condition", "test_snr_db"]
SCORE_COLUMNS: list[str] = [
    *RUN_KEYS,
    "test_snr_db",
    "machine_type",
    "machine_id",
    "label",
    "clip",
    "y",
    "strict_exclude",
    "score",
]


def load_scores(
    runs_dir: Path, split: str, models: Iterable[str] = MODELS
) -> pd.DataFrame:
    """Concatenates the per-clip scores of the runs of a split.

    Args:
        runs_dir: ``<output_root>/runs``.
        split: ``main`` or ``sensitivity``.
        models: Models to load.

    Returns:
        One row per (run, test clip, test SNR), columns :data:`SCORE_COLUMNS`.
    """
    parts = []
    for model in models:
        for path in sorted((runs_dir / split / model).glob("*/seed*/scores.csv")):
            parts.append(
                pd.read_csv(
                    path,
                    usecols=SCORE_COLUMNS,
                    dtype={"machine_id": str, "clip": str},
                )
            )
    if not parts:
        raise FileNotFoundError(f"no scores under {runs_dir / split}")
    return pd.concat(parts, ignore_index=True)[SCORE_COLUMNS]


def _per_run(job: tuple[tuple, pd.DataFrame, float, dict | None, bool]) -> pd.DataFrame:
    keys, part, max_fpr, bootstrap, strict = job
    table = per_id_table(part, max_fpr, strict, bootstrap)
    for position, (name, value) in enumerate(zip(RUN_KEYS, keys, strict=True)):
        table.insert(position, name, value)
    return table


def per_id(
    scores: pd.DataFrame,
    max_fpr: float,
    bootstrap: dict[str, Any] | None = None,
    strict: bool = False,
    workers: int = 1,
) -> pd.DataFrame:
    """AUC and pAUC per (run, test SNR, machine ID), optionally with intervals.

    Args:
        scores: Output of :func:`load_scores`.
        max_fpr: pAUC bound.
        bootstrap: ``evaluation.bootstrap`` of the protocol; None to skip.
        strict: Strict identity variant (flagged rows removed).
        workers: Processes used for the bootstrap (one run per task).

    Returns:
        One row per (model, training condition, seed, test SNR, machine type,
        machine ID).
    """
    jobs = [
        (keys, part, max_fpr, bootstrap, strict)
        for keys, part in scores.groupby(RUN_KEYS, sort=True)
    ]
    if workers > 1:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(workers, mp_context=context) as pool:
            tables = list(pool.map(_per_run, jobs))
    else:
        tables = [_per_run(job) for job in jobs]
    return pd.concat(tables, ignore_index=True)


def aggregated(
    per_id_results: pd.DataFrame,
    value: str = "auc",
    drop: Iterable[Mapping[str, str]] = (),
) -> pd.DataFrame:
    """Aggregated value per (run, test SNR): mean over IDs, then machine types.

    Args:
        per_id_results: Output of :func:`per_id`.
        value: ``auc`` or ``pauc``.
        drop: Machine IDs left out (robustness aggregations).

    Returns:
        Columns ``model, train_condition, seed, test_snr_db, <value>``.
    """
    kept = drop_ids(per_id_results, drop)
    rows = []
    for keys, part in kept.groupby([*RUN_KEYS, "test_snr_db"], sort=True):
        rows.append(
            dict(zip([*RUN_KEYS, "test_snr_db"], keys, strict=True))
            | {value: aggregate(part, value)[1]}
        )
    return pd.DataFrame(rows)


def over_seeds(per_seed: pd.DataFrame, value: str, keys: list[str]) -> pd.DataFrame:
    """Mean and sample standard deviation over seeds (protocol, ddof = 1).

    Args:
        per_seed: One row per seed and cell, with ``seed`` and ``value``.
        value: Column to summarize.
        keys: Columns identifying a cell.

    Returns:
        One row per cell: ``mean``, ``std``, ``n_seeds`` and the per-seed values.
    """
    rows = []
    for cell, part in per_seed.groupby(keys, sort=False):
        part = part.sort_values("seed")
        mean, std = seed_summary(part[value])
        rows.append(
            dict(zip(keys, cell if isinstance(cell, tuple) else (cell,), strict=True))
            | {
                "mean": mean,
                "std": std,
                "n_seeds": len(part),
                "per_seed": [float(v) for v in part[value]],
            }
        )
    return pd.DataFrame(rows)


def matched_condition(test_snr: int) -> str:
    """Training condition whose SNR is ``test_snr``."""
    for condition, snr in CONDITION_SNR.items():
        if snr == test_snr:
            return condition
    raise ValueError(f"no single-SNR condition at {test_snr} dB")


def _paired(agg: pd.DataFrame, value: str) -> pd.DataFrame:
    """Each mismatched or multi-condition row next to the matched run of its seed."""
    single = agg["train_condition"].map(CONDITION_SNR)
    matched = agg[single == agg["test_snr_db"]][
        ["model", "seed", "test_snr_db", value]
    ].rename(columns={value: "matched"})
    paired = agg[single != agg["test_snr_db"]].merge(
        matched, on=["model", "seed", "test_snr_db"], how="left"
    )
    if paired["matched"].isna().any():
        raise ValueError("a matched run is missing for some seed")
    return paired


def deltas(agg: pd.DataFrame, value: str = "auc") -> pd.DataFrame:
    """ΔAUC per seed: value(train c, test s') - value(train s', test s').

    Args:
        agg: Output of :func:`aggregated` (one split, several conditions).
        value: ``auc`` or ``pauc``.

    Returns:
        One row per (model, mismatched or multi condition, seed, test SNR) with
        ``delta``; the matched cells (ΔAUC = 0 by definition) are left out.
    """
    paired = _paired(agg, value)
    return paired.assign(delta=paired[value] - paired["matched"])[
        [*RUN_KEYS, "test_snr_db", value, "matched", "delta"]
    ]


def error_ratios(agg: pd.DataFrame, value: str = "auc") -> pd.DataFrame:
    """Ratio of errors per seed: (1 - value(train c, test s')) / (1 - value(train s', test s')).

    Computed on the aggregated values (a ratio of aggregates, not a mean of
    per-ID ratios). Secondary analysis decided after the stage-A results
    (protocol, "Analyses added after the results").

    Args:
        agg: Output of :func:`aggregated`.
        value: ``auc`` or ``pauc``.

    Returns:
        As :func:`deltas`, with ``ratio`` instead of ``delta``.
    """
    paired = _paired(agg, value)
    return paired.assign(ratio=(1.0 - paired[value]) / (1.0 - paired["matched"]))[
        [*RUN_KEYS, "test_snr_db", value, "matched", "ratio"]
    ]


# ------------------------------------------------- paired bootstrap (post hoc)

IDENTITY_INDEX: list[str] = ["machine_type", "machine_id", "label", "clip"]
COLUMN_KEYS: list[str] = [*RUN_KEYS, "test_snr_db"]


def score_matrix(scores: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Scores with one row per test identity and one column per (run, test SNR).

    Args:
        scores: Output of :func:`load_scores` (one split).

    Returns:
        ``(matrix, y)``: the matrix (rows sorted by identity) and the labels of
        its rows (1 = anomalous).

    Raises:
        ValueError: If some run lacks a test identity at some test SNR.
    """
    matrix = scores.pivot(index=IDENTITY_INDEX, columns=COLUMN_KEYS, values="score")
    if matrix.isna().to_numpy().any():
        raise ValueError("some run lacks a test identity at some test SNR")
    labels = scores.drop_duplicates(IDENTITY_INDEX).set_index(IDENTITY_INDEX)["y"]
    return matrix, labels.reindex(matrix.index).to_numpy()


def rank_auc(values: np.ndarray, y: np.ndarray) -> np.ndarray:
    """AUC of every column (Mann-Whitney with mid-ranks, equal to ``roc_auc_score``).

    Args:
        values: Scores, shape ``(n_clips, n_columns)``.
        y: Labels of the rows (1 = anomalous).

    Returns:
        Array ``(n_columns,)``.
    """
    positive = y == 1
    n_pos = int(positive.sum())
    n_neg = len(y) - n_pos
    ranks = rankdata(values, axis=0)
    return (ranks[positive].sum(axis=0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _bootstrap_group(
    job: tuple[np.ndarray, np.ndarray, int, list[int], float | None],
) -> dict[str, np.ndarray]:
    values, y, n_resamples, seed, max_fpr = job
    rng = np.random.default_rng(seed)
    negatives = np.flatnonzero(y == 0)
    positives = np.flatnonzero(y == 1)
    out = {"auc": np.empty((n_resamples, values.shape[1]))}
    if max_fpr is not None:
        out["pauc"] = np.empty((n_resamples, values.shape[1]))
    for b in range(n_resamples):
        idx = np.concatenate(
            [
                rng.choice(negatives, size=len(negatives), replace=True),
                rng.choice(positives, size=len(positives), replace=True),
            ]
        )
        out["auc"][b] = rank_auc(values[idx], y[idx])
        if max_fpr is not None:
            out["pauc"][b] = [
                pauc(y[idx], values[idx, j], max_fpr) for j in range(values.shape[1])
            ]
    return out


@dataclass
class PairedResamples:
    """Per-ID metrics on the full test set and on paired resamples.

    Attributes:
        columns: (model, train_condition, seed, test_snr_db) of each column.
        groups: (machine type, machine ID) of each row of the arrays.
        point: Metric -> array ``(n_ids, n_columns)`` on the full test set.
        boot: Metric -> array ``(n_ids, n_resamples, n_columns)``.
    """

    columns: pd.MultiIndex
    groups: list[tuple[str, str]]
    point: dict[str, np.ndarray]
    boot: dict[str, np.ndarray]

    def aggregate(
        self, metric: str = "auc", drop: Iterable[Mapping[str, str]] = ()
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Protocol aggregation (mean over IDs, then types), optionally without IDs.

        Args:
            metric: ``auc`` or ``pauc``.
            drop: Machine IDs left out (robustness aggregations).

        Returns:
            ``(point, boot)``: one row and one row per resample.
        """
        dropped = {(d["machine_type"], d["machine_id"]) for d in drop}
        keep = np.array([g not in dropped for g in self.groups])
        types = np.array([t for t, _ in self.groups])[keep]

        def by_type(per_id_values: np.ndarray) -> np.ndarray:
            kept = per_id_values[keep]
            return np.mean(
                [kept[types == t].mean(axis=0) for t in sorted(set(types))], axis=0
            )

        point = pd.DataFrame(by_type(self.point[metric])[None, :], columns=self.columns)
        boot = pd.DataFrame(by_type(self.boot[metric]), columns=self.columns)
        return point, boot

    def machine_type(
        self, metric: str, machine_type: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Mean over the IDs of one machine type (per-type values).

        Args:
            metric: ``auc`` or ``pauc``.
            machine_type: Machine type.

        Returns:
            ``(point, boot)``: one row and one row per resample.
        """
        keep = np.array([t == machine_type for t, _ in self.groups])
        point = pd.DataFrame(
            self.point[metric][keep].mean(axis=0)[None, :], columns=self.columns
        )
        boot = pd.DataFrame(self.boot[metric][keep].mean(axis=0), columns=self.columns)
        return point, boot


def resample_ids(
    scores: pd.DataFrame,
    n_resamples: int,
    seed: int,
    workers: int = 1,
    max_fpr: float | None = None,
) -> PairedResamples:
    """Per-ID AUC (and pAUC) on the full test set and on paired resamples.

    For each machine ID, its test identities are resampled with replacement,
    normal and anomalous separately (counts kept), with one generator per machine
    ID (seeded by ``seed`` and the machine ID). The same draws apply to every
    column: every model, training condition, seed and test SNR, so that
    differences between conditions and between models are paired. Machine IDs
    are not resampled.

    Args:
        scores: Output of :func:`load_scores` (one split, one evaluation).
        n_resamples: Number of resamples.
        seed: Seed of the resampling.
        workers: Processes (one machine ID per task).
        max_fpr: Also compute the pAUC with this bound (None: AUC only).

    Returns:
        The per-ID values, to be aggregated with :meth:`PairedResamples.aggregate`.
    """
    matrix, y = score_matrix(scores)
    keys = matrix.index.droplevel(["label", "clip"])
    groups = sorted(set(keys))
    rows = [np.flatnonzero(keys == group) for group in groups]
    values = matrix.to_numpy()
    jobs = [
        (
            values[r],
            y[r],
            n_resamples,
            [seed, zlib.crc32(f"{machine_type}/{machine_id}".encode())],
            max_fpr,
        )
        for (machine_type, machine_id), r in zip(groups, rows, strict=True)
    ]
    if workers > 1:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(workers, mp_context=context) as pool:
            boot_ids = list(pool.map(_bootstrap_group, jobs))
    else:
        boot_ids = [_bootstrap_group(job) for job in jobs]
    point = {"auc": np.stack([rank_auc(values[r], y[r]) for r in rows])}
    if max_fpr is not None:
        point["pauc"] = np.stack(
            [
                [pauc(y[r], values[r, j], max_fpr) for j in range(values.shape[1])]
                for r in rows
            ]
        )
    boot = {m: np.stack([b[m] for b in boot_ids]) for m in point}
    return PairedResamples(matrix.columns, groups, point, boot)


def paired_bootstrap(
    scores: pd.DataFrame, n_resamples: int, seed: int, workers: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregated AUC per (run, test SNR) on the full test set and on paired resamples.

    Shorthand for ``resample_ids(...).aggregate("auc")``.

    Args:
        scores: Output of :func:`load_scores` (one split).
        n_resamples: Number of resamples.
        seed: Seed of the resampling.
        workers: Processes (one machine ID per task).

    Returns:
        ``(point, boot)``: one row (full test set) and ``n_resamples`` rows; columns
        indexed by (model, train_condition, seed, test_snr_db).
    """
    return resample_ids(scores, n_resamples, seed, workers).aggregate("auc")


def mean_over_seeds(
    frame: pd.DataFrame, model: str, condition: str, test_snr: int
) -> np.ndarray:
    """Aggregated value of one cell averaged over seeds, for each row of ``frame``."""
    return _cell(frame, model, condition, test_snr).mean(axis=1).to_numpy()


def _cell(
    frame: pd.DataFrame, model: str, condition: str, test_snr: int
) -> pd.DataFrame:
    """Columns of one cell (one per seed), rows = samples."""
    return frame.xs(
        (model, condition, test_snr),
        axis=1,
        level=["model", "train_condition", "test_snr_db"],
    )


def delta_over_seeds(
    frame: pd.DataFrame, model: str, condition: str, test_snr: int
) -> np.ndarray:
    """Seed-paired ΔAUC averaged over seeds, for each row of ``frame``."""
    mismatched = _cell(frame, model, condition, test_snr)
    matched = _cell(frame, model, matched_condition(test_snr), test_snr)
    return (mismatched - matched[mismatched.columns]).mean(axis=1).to_numpy()


def ratio_over_seeds(
    frame: pd.DataFrame, model: str, condition: str, test_snr: int
) -> np.ndarray:
    """Ratio of errors per seed averaged over seeds, for each row of ``frame``."""
    mismatched = _cell(frame, model, condition, test_snr)
    matched = _cell(frame, model, matched_condition(test_snr), test_snr)
    return (
        ((1.0 - mismatched) / (1.0 - matched[mismatched.columns]))
        .mean(axis=1)
        .to_numpy()
    )


def interval(point: float, samples: np.ndarray, confidence: float) -> dict[str, float]:
    """Point estimate and percentile interval of a bootstrap statistic."""
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(samples, [alpha, 1.0 - alpha])
    return {"estimate": float(point), "low": float(low), "high": float(high)}
