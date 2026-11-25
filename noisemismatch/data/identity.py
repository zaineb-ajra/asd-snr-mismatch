"""Content-based clip identity: acceptance rule applied to score matrices.

Implements the rule of docs/protocol.md ("Clip identity"). Score matrices are
computed by :mod:`noisemismatch.data.matching`; this module only decides which
(reference, file) pairs are accepted, which identities are kept, and summarizes
the validation of the rule.

Rows of a score matrix are 0 dB reference files (candidate identities); columns
are the files of the same machine type at the compared SNR.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import binom

from noisemismatch.config import CONFIG_DIR, load_paths, load_yaml
from noisemismatch.data.download import LOG_SUBDIR
from noisemismatch.data.matching import MATCHING_SUBDIR
from noisemismatch.data.prepare import METADATA_DIR
from noisemismatch.utils.logs import setup_logging

logger = logging.getLogger(__name__)

IDENTITY_PATH: Path = METADATA_DIR / "mimii_identity.csv"

UNASSIGNED: str = "unassigned"
NOT_MUTUAL: str = "not_mutual_best"
LOW_SCORE: str = "low_score"
LOW_MARGIN: str = "low_margin"
LOW_SUPPORT: str = "low_support"
NO_P6_MATCH: str = "no_p6_match"
OK: str = "ok"


@dataclass(frozen=True)
class MatchRule:
    """Thresholds of the acceptance rule.

    Attributes:
        min_score: Minimum score of an accepted pair (tau).
        min_margin: Minimum difference between the pair's score and the best
            competing score in its row and in its column (delta).
        min_support: Minimum value of each support matrix at the pair (used at
            -6 dB); ignored when no support matrix is given.
    """

    min_score: float
    min_margin: float
    min_support: float = 0.0


def _top_two(
    matrix: np.ndarray, axis: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns the argmax, maximum and second maximum along ``axis``."""
    if matrix.shape[axis] < 2:
        raise ValueError("need at least two candidates per row and column")
    order = np.argsort(matrix, axis=axis)
    first = np.take(order, -1, axis=axis)
    top1 = np.take_along_axis(matrix, np.expand_dims(first, axis), axis=axis)
    second = np.take(order, -2, axis=axis)
    top2 = np.take_along_axis(matrix, np.expand_dims(second, axis), axis=axis)
    return first, top1.squeeze(axis), top2.squeeze(axis)


def accept_matches(
    score: np.ndarray,
    rule: MatchRule,
    support: tuple[np.ndarray, ...] = (),
) -> pd.DataFrame:
    """Applies the acceptance rule to a score matrix.

    A row i assigned to column j by the one-to-one assignment maximizing the
    total score is accepted if (i, j) is a mutual best match, its score is at
    least ``rule.min_score``, its margin over the best competitor in row i and in
    column j is at least ``rule.min_margin``, and every support matrix is at least
    ``rule.min_support`` at (i, j).

    Args:
        score: Matrix shaped ``(n_rows, n_cols)``, at least 2 x 2.
        rule: Thresholds.
        support: Optional matrices of the same shape as ``score``.

    Returns:
        One row per row of ``score`` with columns ``col`` (assigned column or
        -1), ``score``, ``margin``, ``mutual_best``, ``accepted`` and ``reason``
        (``"ok"`` or the first failed criterion, in the order above).
    """
    n_rows = score.shape[0]
    row_arg, row_top1, row_top2 = _top_two(score, axis=1)
    col_arg, col_top1, col_top2 = _top_two(score, axis=0)
    assigned = np.full(n_rows, -1)
    rows, cols = linear_sum_assignment(score, maximize=True)
    assigned[rows] = cols

    records = []
    for i in range(n_rows):
        j = int(assigned[i])
        if j < 0:
            records.append((j, np.nan, np.nan, False, False, UNASSIGNED))
            continue
        value = float(score[i, j])
        row_other = row_top2[i] if row_arg[i] == j else row_top1[i]
        col_other = col_top2[j] if col_arg[j] == i else col_top1[j]
        margin = value - float(max(row_other, col_other))
        mutual = bool(row_arg[i] == j and col_arg[j] == i)
        if not mutual:
            reason = NOT_MUTUAL
        elif value < rule.min_score:
            reason = LOW_SCORE
        elif margin < rule.min_margin:
            reason = LOW_MARGIN
        elif any(float(s[i, j]) < rule.min_support for s in support):
            reason = LOW_SUPPORT
        else:
            reason = OK
        records.append((j, value, margin, mutual, reason == OK, reason))
    return pd.DataFrame(
        records,
        columns=["col", "score", "margin", "mutual_best", "accepted", "reason"],
    )


def match_identities(
    s_p6: np.ndarray,
    s_0m: np.ndarray,
    s_6m: np.ndarray,
    rule: MatchRule,
) -> pd.DataFrame:
    """Matches 0 dB references to +6 dB then -6 dB files.

    Args:
        s_p6: Scores between 0 dB references (rows) and +6 dB files (columns).
        s_0m: Scores between 0 dB references and -6 dB files.
        s_6m: Scores between +6 dB files (rows) and -6 dB files.
        rule: Thresholds; ``min_support`` applies to the two -6 dB terms.

    Returns:
        One row per 0 dB reference with, for ``p6`` and ``m6``: the matched file
        column (``*_col``, -1 if none), ``*_score``, ``*_margin``,
        ``*_accepted`` and ``*_reason``; plus ``m6_term0`` and ``m6_term6``
        (the two terms of the -6 dB score), ``m6_argmax_agree`` (both terms
        maximized by the same -6 dB file) and ``kept``.
    """
    n_refs = s_p6.shape[0]
    p6 = accept_matches(s_p6, rule)
    result = pd.DataFrame(index=range(n_refs))
    for column in ("col", "score", "margin", "accepted", "reason"):
        result[f"p6_{column}"] = p6[column].to_numpy()

    result["m6_col"] = -1
    result["m6_score"] = np.nan
    result["m6_margin"] = np.nan
    result["m6_term0"] = np.nan
    result["m6_term6"] = np.nan
    result["m6_accepted"] = False
    result["m6_reason"] = NO_P6_MATCH
    result["m6_argmax_agree"] = False

    refs = np.flatnonzero(result["p6_accepted"].to_numpy())
    if len(refs) >= 2:
        pi = result.loc[refs, "p6_col"].to_numpy().astype(int)
        term0, term6 = s_0m[refs], s_6m[pi]
        m6 = accept_matches((term0 + term6) / 2, rule, support=(term0, term6))
        cols = m6["col"].to_numpy()
        has = cols >= 0
        result.loc[refs, "m6_col"] = cols
        result.loc[refs, "m6_score"] = m6["score"].to_numpy()
        result.loc[refs, "m6_margin"] = m6["margin"].to_numpy()
        result.loc[refs, "m6_accepted"] = m6["accepted"].to_numpy()
        result.loc[refs, "m6_reason"] = m6["reason"].to_numpy()
        safe = np.where(has, cols, 0)
        rows = np.arange(len(refs))
        result.loc[refs, "m6_term0"] = np.where(has, term0[rows, safe], np.nan)
        result.loc[refs, "m6_term6"] = np.where(has, term6[rows, safe], np.nan)
        agree = (term0.argmax(axis=1) == cols) & (term6.argmax(axis=1) == cols)
        result.loc[refs, "m6_argmax_agree"] = agree & has
    result["kept"] = result["p6_accepted"] & result["m6_accepted"]
    return result


FILE_KEYS: list[str] = ["machine_id", "label", "clip"]
DUPLICATE_THRESHOLD: float = 0.5
NUMERIC_COLUMNS: tuple[str, ...] = (
    "p6_score",
    "p6_margin",
    "m6_score",
    "m6_margin",
    "m6_term0",
    "m6_term6",
)
BOOLEAN_COLUMNS: tuple[str, ...] = (
    "kept",
    "p6_accepted",
    "m6_accepted",
    "m6_argmax_agree",
    "p6_same_name",
    "m6_same_name",
    "p6_cross_id",
    "m6_cross_id",
    "p6_cross_label",
    "m6_cross_label",
    "p6_below_null_p999",
    "m6_below_null_p999",
    "p6_sign_exceeded",
    "m6_sign_exceeded",
)


def identity_table(
    matching_dir: Path, machine_type: str, rule: MatchRule
) -> pd.DataFrame:
    """Applies the rule to the saved score matrices of one machine type.

    Args:
        matching_dir: Output directory of :mod:`noisemismatch.data.matching`.
        machine_type: Machine type.
        rule: Thresholds.

    Returns:
        One row per 0 dB reference (candidate identity): identity columns, the
        matched +6 and -6 dB files (``p6_*`` / ``m6_*`` machine ID, label and
        clip, empty when unassigned), the outputs of :func:`match_identities`,
        and flags telling whether each match keeps the same name, ID and label.
    """
    files = {
        snr: pd.read_csv(matching_dir / f"{machine_type}_files_{snr}.csv", dtype=str)
        for snr in (0, 6, -6)
    }
    scores = np.load(matching_dir / f"{machine_type}_scores.npz")
    result = match_identities(scores["s_p6"], scores["s_0m"], scores["s_6m"], rule)
    table = files[0][FILE_KEYS].copy()
    table.insert(0, "machine_type", machine_type)
    for prefix, snr in (("p6", 6), ("m6", -6)):
        cols = result[f"{prefix}_col"].to_numpy().astype(int)
        matched = files[snr][FILE_KEYS].reindex(cols).reset_index(drop=True)
        matched[cols < 0] = ""
        for key in FILE_KEYS:
            table[f"{prefix}_{key}"] = matched[key].fillna("").to_numpy()
        has = cols >= 0
        table[f"{prefix}_same_name"] = has & np.all(
            [table[k] == table[f"{prefix}_{k}"] for k in FILE_KEYS], axis=0
        )
        table[f"{prefix}_cross_id"] = has & (
            table["machine_id"] != table[f"{prefix}_machine_id"]
        )
        table[f"{prefix}_cross_label"] = has & (
            table["label"] != table[f"{prefix}_label"]
        )
    for column in result.columns:
        if not column.endswith("_col"):
            table[column] = result[column].to_numpy()
    return table


def duplicate_pairs(matching_dir: Path, machine_type: str) -> pd.DataFrame:
    """Lists pairs of files of one SNR whose scores reach ``DUPLICATE_THRESHOLD``.

    A score of about 0.8 is expected at +6 dB for two copies of one recording
    with independent noise, and at most about 0.2 for files sharing only a noise
    segment; this is a diagnostic, not part of the identity rule.

    Args:
        matching_dir: Output directory of :mod:`noisemismatch.data.matching`.
        machine_type: Machine type.

    Returns:
        One row per pair (``snr_db``, both files, ``score``).
    """
    scores = np.load(matching_dir / f"{machine_type}_scores.npz")
    rows = []
    for snr, name in ((0, "s_00"), (6, "s_66"), (-6, "s_mm")):
        files = pd.read_csv(matching_dir / f"{machine_type}_files_{snr}.csv", dtype=str)
        matrix = scores[name]
        i, k = np.nonzero(np.triu(matrix >= DUPLICATE_THRESHOLD, k=1))
        for a, b in zip(i, k, strict=True):
            rows.append(
                {
                    "machine_type": machine_type,
                    "snr_db": snr,
                    **{f"a_{c}": files.loc[a, c] for c in FILE_KEYS},
                    **{f"b_{c}": files.loc[b, c] for c in FILE_KEYS},
                    "score": float(matrix[a, b]),
                }
            )
    return pd.DataFrame(rows)


def load_identity(path: Path = IDENTITY_PATH) -> pd.DataFrame:
    """Reads the identity table with identifiers as strings and flags as booleans.

    Args:
        path: CSV written by :mod:`noisemismatch.data.identity`.

    Returns:
        The identity table.
    """
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in BOOLEAN_COLUMNS:
        if column in table:
            table[column] = table[column] == "True"
    for column in NUMERIC_COLUMNS:
        table[column] = pd.to_numeric(table[column].replace("", None))
    return table


def file_positions(matching_dir: Path, machine_type: str, snr: int) -> dict:
    """Maps (machine ID, label, clip) to the row of a file in the score matrices."""
    files = pd.read_csv(matching_dir / f"{machine_type}_files_{snr}.csv", dtype=str)
    return {tuple(row): k for k, row in enumerate(files[FILE_KEYS].to_numpy())}


def _accepted_rows(
    identity: pd.DataFrame, matching_dir: Path
) -> Iterator[tuple[int, str, pd.DataFrame, np.ndarray, np.ndarray, tuple | None]]:
    """Yields, per (machine type, SNR, machine ID, label), accepted rows and scores.

    Yields:
        ``(snr, prefix, rows, matrix, cols, terms)``: ``rows`` are the accepted
        identity rows of the group, ``matrix`` their score rows over all files of
        the machine type (the rule's score: at -6 dB the average of the two
        terms), ``cols`` the accepted file of each row, and ``terms`` the two
        -6 dB score terms (None at +6 dB).
    """
    for machine_type, table in identity.groupby("machine_type"):
        scores = np.load(matching_dir / f"{machine_type}_scores.npz")
        table = table.reset_index(drop=True)  # row order of the score matrices
        position = {
            snr: file_positions(matching_dir, machine_type, snr) for snr in (6, -6)
        }
        for snr, prefix, mask in (
            (6, "p6", table["p6_accepted"]),
            (-6, "m6", table["kept"]),
        ):
            for _, rows in table[mask].groupby(["machine_id", "label"]):
                i = rows.index.to_numpy()
                p6 = rows[[f"p6_{k}" for k in FILE_KEYS]].to_numpy()
                j6 = np.array([position[6][tuple(r)] for r in p6])
                if snr == 6:
                    yield snr, prefix, rows, scores["s_p6"][i], j6, None
                else:
                    terms = (scores["s_0m"][i], scores["s_6m"][j6])
                    m6 = rows[[f"m6_{k}" for k in FILE_KEYS]].to_numpy()
                    cols = np.array([position[-6][tuple(r)] for r in m6])
                    yield snr, prefix, rows, (terms[0] + terms[1]) / 2, cols, terms


def _row_statistics(matrix: np.ndarray, cols: np.ndarray) -> dict[str, np.ndarray]:
    """Accepted score, best and most negative non-match of each row."""
    r = np.arange(len(cols))
    accepted = matrix[r, cols].copy()
    others = matrix.copy()
    others[r, cols] = np.nan
    return {
        "accepted": accepted,
        "best_other": np.nanmax(others, axis=1),
        "most_negative": np.nanmin(others, axis=1),
        "null": others[~np.isnan(others)],
    }


def symmetry_table(
    identity: pd.DataFrame, matching_dir: Path, rule: MatchRule
) -> pd.DataFrame:
    """Compares accepted matches with the non-matches of their rows.

    For every accepted identity (+6 dB: accepted +6 dB match; -6 dB: kept), the
    non-matches are all other files of the machine type in the identity's row of
    the score matrix (-6 dB: the averaged score of the rule). Coincidental
    correlations of tonal sounds are sign-symmetric, whereas true matches are
    positive. Under the null hypothesis that accepted matches are such
    coincidences, each accepted score is exceeded by the most negative
    non-match of its row (in absolute value) with probability 1/2; the
    one-sided binomial p-value of the observed number of exceedances is
    reported, and also with that probability calibrated as the rate at which
    each row's best non-match is exceeded by the row's most negative one.

    ``strong_second`` counts the rows whose best non-match would itself pass
    the rule's row-wise thresholds (score, margin over the next non-match and,
    at -6 dB, support of both terms); the accepted match then beat it by the
    margin.

    Args:
        identity: Identity table (:func:`load_identity`).
        matching_dir: Directory of the score matrices.
        rule: Thresholds of the identity rule.

    Returns:
        One row per (SNR, machine type, machine ID, label).
    """
    records = []
    for snr, prefix, rows, matrix, cols, terms in _accepted_rows(
        identity, matching_dir
    ):
        stats = _row_statistics(matrix, cols)
        r = np.arange(len(cols))
        others = matrix.copy()
        others[r, cols] = -np.inf
        best_col = others.argmax(axis=1)
        best = others[r, best_col]
        others[r, best_col] = -np.inf
        strong = (best >= rule.min_score) & (
            best - others.max(axis=1) >= rule.min_margin
        )
        if terms is not None:
            strong &= (terms[0][r, best_col] >= rule.min_support) & (
                terms[1][r, best_col] >= rule.min_support
            )
        accepted, null = stats["accepted"], stats["null"]
        negative = -stats["most_negative"]
        median = float(np.median(accepted))
        p999 = float(np.quantile(np.abs(null), 0.999))
        other = ~rows[f"{prefix}_same_name"].to_numpy()
        exceeded = int((negative >= accepted).sum())
        q_calibrated = float((negative >= stats["best_other"]).mean())
        first = rows.iloc[0]
        records.append(
            {
                "snr_db": snr,
                "machine_type": first["machine_type"],
                "machine_id": first["machine_id"],
                "label": first["label"],
                "accepted": len(rows),
                "other_name": int(other.sum()),
                "median_accepted": median,
                "median_accepted_other_name": (
                    float(np.median(accepted[other])) if other.any() else np.nan
                ),
                "null_p999_abs": p999,
                "null_min": float(null.min()),
                "rows_with_null_below_minus_median": int((-negative <= -median).sum()),
                "accepted_below_p999": int((accepted < p999).sum()),
                "strong_second": int(strong.sum()),
                "sign_exceeded": exceeded,
                "p_binomial_half": float(binom.cdf(exceeded, len(rows), 0.5)),
                "q_calibrated": q_calibrated,
                "p_binomial_calibrated": float(
                    binom.cdf(exceeded, len(rows), q_calibrated)
                ),
            }
        )
    return pd.DataFrame(records)


def null_flags(identity: pd.DataFrame, matching_dir: Path) -> pd.DataFrame:
    """Per-identity comparison of the accepted score with its group's null.

    Args:
        identity: Identity table (:func:`load_identity`).
        matching_dir: Directory of the score matrices.

    Returns:
        ``identity`` with boolean columns ``p6_below_null_p999`` and
        ``m6_below_null_p999`` (accepted score below the 99.9th percentile of
        |score| over the non-matches of the group, see :func:`symmetry_table`)
        and ``p6_sign_exceeded`` / ``m6_sign_exceeded`` (accepted score not above
        the most negative non-match of its row, in absolute value).
    """
    out = identity.copy()
    for prefix in ("p6", "m6"):
        out[f"{prefix}_below_null_p999"] = False
        out[f"{prefix}_sign_exceeded"] = False
    keys = ["machine_type", "machine_id", "label", "clip"]
    position = {tuple(k): n for n, k in enumerate(out[keys].to_numpy())}
    for _, prefix, rows, matrix, cols, _ in _accepted_rows(identity, matching_dir):
        stats = _row_statistics(matrix, cols)
        p999 = float(np.quantile(np.abs(stats["null"]), 0.999))
        index = [position[tuple(k)] for k in rows[keys].to_numpy()]
        out.loc[index, f"{prefix}_below_null_p999"] = stats["accepted"] < p999
        out.loc[index, f"{prefix}_sign_exceeded"] = (
            -stats["most_negative"] >= stats["accepted"]
        )
    return out


def validation_summary(table: pd.DataFrame) -> dict[str, float]:
    """Validation figures of the rule.

    Between 0 and +6 dB, the same-name file is taken as ground truth.

    Args:
        table: Concatenated outputs of :func:`identity_table`.

    Returns:
        ``p6_precision`` (accepted +6 dB pairs whose names agree), ``p6_recall``
        (references accepted with their same-name file), ``m6_argmax_agreement``
        (accepted -6 dB pairs on which both score terms agree) and counts.
    """
    accepted = table["p6_accepted"]
    good = accepted & table["p6_same_name"]
    m6 = table["m6_accepted"]
    return {
        "n_references": len(table),
        "p6_accepted": int(accepted.sum()),
        "p6_precision": float(good.sum() / accepted.sum()),
        "p6_recall": float(good.sum() / len(table)),
        "m6_accepted": int(m6.sum()),
        "m6_argmax_agreement": float(table.loc[m6, "m6_argmax_agree"].mean()),
        "kept": int(table["kept"].sum()),
    }


def main(argv: list[str] | None = None) -> int:
    """Builds the identity table of every machine type and logs the summaries.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code (0).
    """
    parser = argparse.ArgumentParser(description="Apply the clip identity rule.")
    parser.parse_args(argv)
    paths = load_paths()
    setup_logging(paths.data_root / LOG_SUBDIR / "identity.log")
    config = load_yaml(CONFIG_DIR / "protocol.yaml")["identity"]
    rule = MatchRule(
        min_score=float(config["min_score"]),
        min_margin=float(config["min_margin"]),
        min_support=float(config["min_support_m6"]),
    )
    matching_dir = paths.data_root / MATCHING_SUBDIR
    types = sorted(
        p.name.removesuffix("_scores.npz") for p in matching_dir.glob("*_scores.npz")
    )
    table = pd.concat(
        [identity_table(matching_dir, t, rule) for t in types], ignore_index=True
    )
    table = null_flags(table, matching_dir)
    table.to_csv(IDENTITY_PATH, index=False, lineterminator="\n", float_format="%.5f")
    duplicates = pd.concat([duplicate_pairs(matching_dir, t) for t in types])
    duplicates.to_csv(
        matching_dir / "duplicate_pairs.csv",
        index=False,
        lineterminator="\n",
        float_format="%.5f",
    )
    logger.info("Rule: %s", rule)
    logger.info("Validation: %s", validation_summary(table))
    logger.info("Identity table written to %s", IDENTITY_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
