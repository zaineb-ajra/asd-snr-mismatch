"""Clip sets: the clips a run trains or tests on, with their audio and features.

A clip table has one row per (clip identity, SNR) with the identity columns, the
SNR, the file that holds the recording at that SNR (content-based identity), the
partition (``split``), the label as an integer ``y`` (1 = anomalous) and
``strict_exclude`` (identity removed in the strict evaluation variant). Rows are
sorted by identity, then SNR, so that runs of the same seed at different SNRs see
their clips in the same order.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from noisemismatch.data.split import (
    ANOMALOUS,
    FILE_COLUMNS,
    IDENTITY_COLUMNS,
    assign_files,
    identity_files,
)
from noisemismatch.features.base import FeatureExtractor
from noisemismatch.features.cache import AudioSource, FeatureCache

ORDER: list[str] = [*IDENTITY_COLUMNS, "snr_db"]
SCOPES: dict[str, list[str]] = {
    "machine_id": ["machine_type", "machine_id"],
    "machine_type": ["machine_type"],
    "all": [],
}


def strict_exclusion(identities: pd.DataFrame, strict: dict[str, Any]) -> pd.Series:
    """Tells which identities the strict evaluation variant removes.

    Args:
        identities: Identity table rows (typed, see ``load_identity``).
        strict: ``evaluation.strict_variant`` of the protocol (lists
            ``exclude_if_false`` and ``exclude_if_true`` of boolean columns).

    Returns:
        Boolean Series aligned with ``identities``.
    """
    exclude = pd.Series(False, index=identities.index)
    for column in strict.get("exclude_if_false", []):
        exclude |= ~identities[column].astype(bool)
    for column in strict.get("exclude_if_true", []):
        exclude |= identities[column].astype(bool)
    return exclude


def build_clip_table(
    split: pd.DataFrame,
    identities: pd.DataFrame,
    snrs: tuple[int, ...],
    strict: dict[str, Any],
) -> pd.DataFrame:
    """Builds the clip table of a split.

    Args:
        split: Split table (identity columns and ``split``).
        identities: Identity table rows of the split's identities.
        snrs: SNRs of the split.
        strict: ``evaluation.strict_variant`` of the protocol.

    Returns:
        The clip table, sorted by identity then SNR.
    """
    files = assign_files(split, identity_files(identities, snrs=snrs, only_kept=False))
    flags = identities[list(IDENTITY_COLUMNS)].copy()
    flags["strict_exclude"] = strict_exclusion(identities, strict).to_numpy()
    table = files.merge(
        flags, on=list(IDENTITY_COLUMNS), how="left", validate="many_to_one"
    )
    table["y"] = (table["label"] == ANOMALOUS).astype(int)
    return table.sort_values(ORDER, ignore_index=True)


def select(
    table: pd.DataFrame,
    partition: str | None = None,
    snrs: tuple[int, ...] | None = None,
    machine_types: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Selects rows of a clip table, keeping its order.

    Args:
        table: Clip table.
        partition: ``train`` or ``test`` (None: both).
        snrs: SNRs to keep (None: all).
        machine_types: Machine types to keep (None: all).

    Returns:
        The selected rows, reindexed.
    """
    mask = pd.Series(True, index=table.index)
    if partition is not None:
        mask &= table["split"] == partition
    if snrs is not None:
        mask &= table["snr_db"].isin(snrs)
    if machine_types is not None:
        mask &= table["machine_type"].isin(machine_types)
    return table[mask].reset_index(drop=True)


@dataclass
class ClipSet:
    """Clips of one training or evaluation step, with access to their data.

    Attributes:
        table: Clip table rows (see module docstring).
        audio: Audio cache.
        cache_root: Root of the feature caches (None: features unavailable).
    """

    table: pd.DataFrame
    audio: AudioSource
    cache_root: Path | None = None

    def __len__(self) -> int:
        return len(self.table)

    @property
    def labels(self) -> np.ndarray:
        """Integer labels (1 = anomalous)."""
        return self.table["y"].to_numpy()

    def _gather(self, load: Any, dtype: type) -> np.ndarray:
        """Loads rows per (machine type, SNR) and reassembles them in table order."""
        out: np.ndarray | None = None
        for (machine_type, snr), part in self.table.groupby(["machine_type", "snr_db"]):
            values = load(machine_type, int(snr), part[list(FILE_COLUMNS)])
            if out is None:
                out = np.empty((len(self.table), *values.shape[1:]), dtype=dtype)
            out[part.index.to_numpy()] = values
        if out is None:
            raise ValueError("empty clip set")
        return out

    def gather(self, load: Any, dtype: type) -> np.ndarray:
        """Values of ``load(machine_type, snr, files)`` reassembled in table order."""
        return self._gather(load, dtype)

    def raw_waveforms(self) -> np.ndarray:
        """Int16 waveforms, shape ``(n_clips, n_samples)`` (half the memory)."""
        return self._gather(self.audio.load, np.int16)

    def waveforms(self) -> np.ndarray:
        """Float32 waveforms in [-1, 1], shape ``(n_clips, n_samples)``."""
        return self.raw_waveforms().astype(np.float32) / 32768.0

    def features(self, extractor: FeatureExtractor) -> np.ndarray:
        """Cached features, shape ``(n_clips, *feature_shape)``.

        Args:
            extractor: Feature extractor; its configuration selects the cache.
        """
        if self.cache_root is None:
            raise ValueError("this clip set has no feature cache root")
        cache = FeatureCache(self.cache_root, extractor, self.audio)
        return self._gather(cache.get, np.float32)

    def subset(self, mask: np.ndarray | pd.Series) -> ClipSet:
        """Rows where ``mask`` is true, reindexed, same audio source."""
        table = self.table[np.asarray(mask)].reset_index(drop=True)
        return ClipSet(table, self.audio, self.cache_root)

    def groups(self, scope: str) -> Iterator[tuple[tuple[str, ...], ClipSet]]:
        """Splits the set by training scope (``machine_id``, ``machine_type``, ``all``).

        Yields:
            ``(group key, clip set)`` in sorted group order.
        """
        keys = SCOPES[scope]
        if not keys:
            yield (), self
            return
        for key, part in self.table.groupby(keys, sort=True):
            key = key if isinstance(key, tuple) else (key,)
            clips = ClipSet(part.reset_index(drop=True), self.audio, self.cache_root)
            yield tuple(key), clips
