"""Common interface of feature extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np


class FeatureExtractor(ABC):
    """Computes a feature array from one waveform.

    Subclasses declare every parameter that affects the output in ``config``; the
    feature cache key is derived from it, so a parameter left out of ``config``
    could make a stale cache look valid.

    Args:
        **config: Extractor parameters (merged over ``defaults``).
    """

    name: ClassVar[str]
    version: ClassVar[int] = 1
    defaults: ClassVar[dict[str, Any]] = {}

    def __init__(self, **config: Any) -> None:
        unknown = sorted(set(config) - set(self.defaults))
        if unknown:
            raise ValueError(f"{self.name}: unknown parameters {unknown}")
        self.config: dict[str, Any] = {**self.defaults, **config}

    @property
    def identity(self) -> dict[str, Any]:
        """Everything that determines the output: name, version and config."""
        return {"name": self.name, "version": self.version, "config": self.config}

    @abstractmethod
    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """Computes the features of one waveform.

        Args:
            waveform: Float32 samples in [-1, 1], shape ``(n_samples,)``.

        Returns:
            Float32 feature array.
        """

    def batch(self, waveforms: np.ndarray) -> np.ndarray:
        """Computes the features of several waveforms (default: one at a time).

        Extractors that run a network override it to process clips in batches.

        Args:
            waveforms: Float32 samples in [-1, 1], shape ``(n_clips, n_samples)``.

        Returns:
            Float32 array shaped ``(n_clips, *feature_shape)``.
        """
        return np.stack([self(w) for w in waveforms])
