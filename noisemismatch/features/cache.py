"""On-disk feature cache keyed by a hash of the feature configuration.

A cache directory is named ``<extractor name>-<key>``, where ``key`` is the SHA-256
of the extractor identity (name, version, full parameter set), the cache format
version and the fingerprint of the source audio cache. Changing any parameter or
the audio therefore selects another directory: a stale cache is never read. The
identity is also stored in the directory (``config.json``) and checked on every
open.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from noisemismatch.data.split import FILE_COLUMNS
from noisemismatch.features.base import FeatureExtractor

logger = logging.getLogger(__name__)

CACHE_FORMAT_VERSION: int = 1
KEY_LENGTH: int = 16
CHUNK: int = 256  # waveforms converted and passed to the extractor at once


class AudioSource(Protocol):
    """What the feature cache needs from the audio cache."""

    @property
    def fingerprint(self) -> str:
        """Identifier of the audio content."""

    def index(self, machine_type: str, snr: int) -> pd.DataFrame:
        """File columns of the rows of one array."""

    def array(self, machine_type: str, snr: int) -> np.ndarray:
        """Int16 waveforms of one machine type and SNR."""

    def load(self, machine_type: str, snr: int, files: pd.DataFrame) -> np.ndarray:
        """Int16 waveforms of the given files, in order."""


def cache_identity(
    extractor: FeatureExtractor, audio_fingerprint: str
) -> dict[str, Any]:
    """Everything that determines the cached features.

    Args:
        extractor: Feature extractor.
        audio_fingerprint: Fingerprint of the source audio cache.

    Returns:
        A JSON-serializable mapping.
    """
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "extractor": extractor.identity,
        "audio": audio_fingerprint,
    }


def cache_key(identity: dict[str, Any]) -> str:
    """Hash of a cache identity (canonical JSON, SHA-256, truncated).

    Args:
        identity: Output of :func:`cache_identity`.

    Returns:
        Hexadecimal key of ``KEY_LENGTH`` characters.
    """
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:KEY_LENGTH]


class FeatureCache:
    """Computes features once per (machine type, SNR) and serves them by file.

    Args:
        cache_root: Root of the caches (``cache_root`` of ``configs/paths.yaml``).
        extractor: Feature extractor.
        audio: Source audio cache.

    Raises:
        ValueError: If the directory exists with a different identity.
    """

    def __init__(
        self, cache_root: Path, extractor: FeatureExtractor, audio: AudioSource
    ) -> None:
        self.extractor = extractor
        self.audio = audio
        self.identity = cache_identity(extractor, audio.fingerprint)
        self.key = cache_key(self.identity)
        self.dir = cache_root / "features" / f"{extractor.name}-{self.key}"
        config_path = self.dir / "config.json"
        stored = {"key": self.key, "identity": self.identity}
        if config_path.exists():
            found = json.loads(config_path.read_text(encoding="utf-8"))
            if found != json.loads(json.dumps(stored)):
                raise ValueError(f"{self.dir}: stored configuration differs")
        else:
            self.dir.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                json.dumps(stored, indent=2) + "\n", encoding="utf-8"
            )

    def _path(self, machine_type: str, snr: int) -> Path:
        return self.dir / f"{machine_type}_{snr}.npy"

    def _compute(self, machine_type: str, snr: int) -> None:
        waveforms = self.audio.array(machine_type, snr)
        features = np.concatenate(
            [
                self.extractor.batch(
                    np.asarray(waveforms[start : start + CHUNK], dtype=np.float32)
                    / 32768.0
                )
                for start in range(0, len(waveforms), CHUNK)
            ]
        )
        tmp = self.dir / f"{machine_type}_{snr}.tmp.npy"
        np.save(tmp, features.astype(np.float32))
        tmp.replace(self._path(machine_type, snr))
        logger.info(
            "%s: features of %s %+d dB cached", self.dir.name, machine_type, snr
        )

    def ensure(self, machine_type: str, snr: int) -> bool:
        """Computes the array of one machine type and SNR if it is missing.

        Returns:
            True if the array was computed, False if it was already cached.
        """
        if self._path(machine_type, snr).exists():
            return False
        self._compute(machine_type, snr)
        return True

    def get(self, machine_type: str, snr: int, files: pd.DataFrame) -> np.ndarray:
        """Returns the features of the given files (computing them if needed).

        Args:
            machine_type: Machine type.
            snr: SNR in dB.
            files: Rows with the file columns, in the order to return.

        Returns:
            Float32 array shaped ``(len(files), *feature_shape)``.
        """
        self.ensure(machine_type, snr)
        index = self.audio.index(machine_type, snr)
        rows = {tuple(r): n for n, r in enumerate(index[list(FILE_COLUMNS)].to_numpy())}
        order = [rows[tuple(r)] for r in files[list(FILE_COLUMNS)].to_numpy()]
        array = np.load(self._path(machine_type, snr), mmap_mode="r")
        return np.asarray(array[order])
