"""Checks that a MIMII clip is the same recording at every SNR.

Two levels:

1. **Names.** Every clip identity (machine type, machine ID, label, clip name)
   must exist at every SNR (:func:`unpaired_identities`).
2. **Content**, on a random sample of clips (:func:`content_pairing`). MIMII
   builds each SNR version as machine recording + scaled factory noise. If the
   three versions of a clip share the machine recording, their waveforms are
   correlated, whereas two different clips are not. The check also fits the
   -6 dB version as a linear combination of the 0 dB and +6 dB versions: a
   near-zero residual means the *same* noise segment was used at every SNR, only
   rescaled (x_s = m + g_s·n spans a 2-D space), which matters when interpreting
   multi-condition training.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from noisemismatch.data.mimii import read_member
from noisemismatch.data.split import IDENTITY_COLUMNS


def unpaired_identities(index: pd.DataFrame) -> pd.DataFrame:
    """Lists clip identities that are missing at some SNR.

    Args:
        index: File index (identity columns and ``snr_db``).

    Returns:
        One row per incomplete identity, with the SNRs where it is present
        (column ``snrs_present``); empty if every identity is complete.
    """
    snrs = sorted(index["snr_db"].unique())
    present = (
        index.groupby(list(IDENTITY_COLUMNS))["snr_db"]
        .agg(lambda s: tuple(sorted(s.unique())))
        .rename("snrs_present")
        .reset_index()
    )
    return present[present["snrs_present"] != tuple(snrs)].reset_index(drop=True)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two 1-D signals."""
    return float(np.corrcoef(a, b)[0, 1])


def _relative_residual(target: np.ndarray, bases: Sequence[np.ndarray]) -> float:
    """Relative least-squares residual of ``target`` on the span of ``bases``."""
    design = np.stack(bases, axis=1)
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    return float(np.linalg.norm(target - design @ coef) / np.linalg.norm(target))


def content_pairing(
    index: pd.DataFrame,
    downloads_dir: Path,
    per_group: int,
    seed: int,
    snrs: tuple[int, int, int] = (-6, 0, 6),
    channel: int = 0,
) -> pd.DataFrame:
    """Compares the SNR versions of a random sample of clips.

    From each (machine type, machine ID, label) group, ``per_group`` clips are
    drawn. For each drawn clip i, with j the next drawn clip of the same group:

    - ``corr_<a>_<b>``: correlation between clip i at SNRs a and b;
    - ``corr_other``: correlation between clip i at the highest SNR and clip j at
      the middle SNR (same SNR pair as ``corr_<mid>_<high>``, different clips);
    - ``residual``: relative residual of clip i at the lowest SNR regressed on
      clip i at the two other SNRs.

    Args:
        index: File index (identity columns, ``snr_db``, ``archive``, ``member``).
        downloads_dir: Directory holding the archives.
        per_group: Number of clips drawn per group (at least 2).
        seed: Seed of the draw.
        snrs: The three SNRs, in increasing order.
        channel: Channel compared.

    Returns:
        One row per drawn clip.
    """
    low, mid, high = snrs
    location = index.set_index([*IDENTITY_COLUMNS, "snr_db"])[["archive", "member"]]

    def load(identity: tuple[str, ...], snr: int) -> np.ndarray:
        archive, member = location.loc[(*identity, snr)]
        samples, _ = read_member(downloads_dir / archive, member, channel=channel)
        return samples.astype(np.float64)

    rng = np.random.default_rng(seed)
    identities = index.loc[:, list(IDENTITY_COLUMNS)].drop_duplicates()
    rows: list[dict[str, object]] = []
    for _, group in identities.sort_values(list(IDENTITY_COLUMNS)).groupby(
        ["machine_type", "machine_id", "label"], sort=True
    ):
        picked = rng.choice(len(group), size=min(per_group, len(group)), replace=False)
        drawn = [tuple(group.iloc[k]) for k in sorted(picked)]
        signals = {ident: {s: load(ident, s) for s in snrs} for ident in drawn}
        for k, ident in enumerate(drawn):
            x = signals[ident]
            other = signals[drawn[(k + 1) % len(drawn)]]
            rows.append(
                {
                    **dict(zip(IDENTITY_COLUMNS, ident, strict=True)),
                    f"corr_{low}_{mid}": _corr(x[low], x[mid]),
                    f"corr_{mid}_{high}": _corr(x[mid], x[high]),
                    f"corr_{low}_{high}": _corr(x[low], x[high]),
                    "corr_other": _corr(x[high], other[mid]),
                    "residual": _relative_residual(x[low], [x[mid], x[high]]),
                }
            )
    return pd.DataFrame(rows)
