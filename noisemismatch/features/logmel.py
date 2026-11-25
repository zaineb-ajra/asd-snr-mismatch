"""Log-mel spectrogram."""

from __future__ import annotations

import sys
from typing import Any, ClassVar

import librosa
import numpy as np

from noisemismatch.features.base import FeatureExtractor


class LogMel(FeatureExtractor):
    """Log-mel power spectrogram, ``20 / power * log10(floor(mel))``.

    The default parameters are those of the MIMII baseline (64 mel bands,
    1024-sample FFT, hop 512, power 2, eps = machine epsilon, ``mel + eps``); the
    DCASE 2022 baseline uses ``max(mel, eps)`` instead (``log_floor="max"``).
    Every librosa option that changes the output is an explicit parameter.

    Returns arrays shaped ``(n_mels, n_frames)``.
    """

    name: ClassVar[str] = "logmel"
    defaults: ClassVar[dict[str, Any]] = {
        "sample_rate": 16_000,
        "n_fft": 1024,
        "hop_length": 512,
        "win_length": None,
        "window": "hann",
        "center": True,
        "pad_mode": "constant",
        "n_mels": 64,
        "fmin": 0.0,
        "fmax": None,
        "htk": False,
        "norm": "slaney",
        "power": 2.0,
        "eps": sys.float_info.epsilon,
        "log_floor": "add",  # "add": mel + eps (MIMII); "max": max(mel, eps) (DCASE)
    }

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """Computes the log-mel spectrogram of one waveform.

        Args:
            waveform: Float32 samples in [-1, 1], shape ``(n_samples,)``.

        Returns:
            Float32 array shaped ``(n_mels, n_frames)``.
        """
        c = self.config
        mel = librosa.feature.melspectrogram(
            y=np.asarray(waveform, dtype=np.float32),
            sr=c["sample_rate"],
            n_fft=c["n_fft"],
            hop_length=c["hop_length"],
            win_length=c["win_length"],
            window=c["window"],
            center=c["center"],
            pad_mode=c["pad_mode"],
            n_mels=c["n_mels"],
            fmin=c["fmin"],
            fmax=c["fmax"],
            htk=c["htk"],
            norm=c["norm"],
            power=c["power"],
        )
        if c["log_floor"] == "add":
            floored = mel + c["eps"]
        elif c["log_floor"] == "max":
            floored = np.maximum(mel, c["eps"])
        else:
            raise ValueError(f"unknown log_floor {c['log_floor']!r}")
        return (20.0 / c["power"] * np.log10(floored)).astype(np.float32)
