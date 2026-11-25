"""Frozen EAT embeddings of a clip (inference only).

Preprocessing of the model card of worstchan/EAT-base_epoch30_pretrain (revision
6eb9593): mean removal, Kaldi filterbank (128 bins, 25 ms Hanning window, 10 ms
shift, HTK-compatible, no dither), zero padding to 1,024 frames, normalization
``(x - mean) / (2 * std)``. The encoder output (CLS + 64 x 8 patch tokens) is
pooled two ways and both are stored, so that the scoring variants of the protocol
share one cache: ``[temporal mean (8 x 768 = 6144), CLS (768)]``.

Computed in float32 with TF32 disabled.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
import torchaudio

from noisemismatch.features.base import FeatureExtractor
from noisemismatch.features.eat_encoder import EMBED_DIM, MEL_BINS, PATCH, load_encoder

SAMPLE_RATE: int = 16000
N_FREQ: int = MEL_BINS // PATCH  # frequency positions of the patch grid (8)
TIME_MEAN_DIM: int = N_FREQ * EMBED_DIM  # 6144
POOLINGS: dict[str, slice] = {
    "time_mean": slice(0, TIME_MEAN_DIM),
    "cls": slice(TIME_MEAN_DIM, TIME_MEAN_DIM + EMBED_DIM),
}


def checkpoint_dir(data_root: Path, repo: str, revision: str) -> Path:
    """Local directory of a pinned Hugging Face snapshot."""
    return data_root / "eat" / repo.split("/")[-1] / revision


def sha256(path: Path) -> str:
    """SHA-256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def disable_tf32() -> None:
    """Full float32 precision for matrix products and convolutions."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def filterbank(
    waveforms: torch.Tensor,
    num_mel_bins: int,
    frame_shift_ms: float,
    target_length: int,
    norm_mean: float,
    norm_std: float,
) -> torch.Tensor:
    """Normalized Kaldi filterbanks of the model card, one clip at a time.

    Args:
        waveforms: Float32 tensor ``(n_clips, n_samples)``, values in [-1, 1].
        num_mel_bins: Mel bins.
        frame_shift_ms: Frame shift in ms.
        target_length: Frames after padding or truncation.
        norm_mean: Normalization mean.
        norm_std: Normalization standard deviation (the input is divided by 2 x).

    Returns:
        Tensor shaped ``(n_clips, 1, target_length, num_mel_bins)``.
    """
    out = []
    for waveform in waveforms:
        waveform = waveform - waveform.mean()
        mel = torchaudio.compliance.kaldi.fbank(
            waveform.unsqueeze(0),
            htk_compat=True,
            sample_frequency=SAMPLE_RATE,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=num_mel_bins,
            dither=0.0,
            frame_shift=frame_shift_ms,
        )
        n_frames = mel.shape[0]
        if n_frames < target_length:
            mel = torch.nn.functional.pad(mel, (0, 0, 0, target_length - n_frames))
        else:
            mel = mel[:target_length]
        out.append((mel - norm_mean) / (norm_std * 2))
    return torch.stack(out).unsqueeze(1)


def pool(tokens: torch.Tensor) -> torch.Tensor:
    """``[temporal mean per frequency position, CLS]`` of encoder outputs.

    Args:
        tokens: Encoder output ``(B, 1 + n_time * 8, 768)``, patches time-major.

    Returns:
        Tensor shaped ``(B, 6144 + 768)``.
    """
    b = tokens.shape[0]
    patches = tokens[:, 1:].reshape(b, -1, N_FREQ, EMBED_DIM)  # (B, time, freq, D)
    return torch.cat([patches.mean(dim=1).reshape(b, -1), tokens[:, 0]], dim=1)


class EatEmbedding(FeatureExtractor):
    """Pooled last-layer EAT embeddings (see module docstring).

    Args:
        device: Torch device for the filterbank and the encoder (not part of the
            cache identity).
        data_root: Root of the downloaded data (holds the checkpoint snapshot).
        **config: Checkpoint and preprocessing parameters.
    """

    name: ClassVar[str] = "eat"
    defaults: ClassVar[dict[str, Any]] = {
        "repo": "worstchan/EAT-base_epoch30_pretrain",
        "revision": "6eb95936bf1061cacee1c0b89404a1aca9a3c62a",
        "file": "model.safetensors",
        "sha256": "8623072d09aac4f3ad1168b4fed3a24e4f68fe1da25b9fe733375efb237e5f48",
        "num_mel_bins": 128,
        "frame_shift_ms": 10,
        "target_length": 1024,
        "norm_mean": -4.268,
        "norm_std": 4.569,
        "batch_size": 32,
    }

    def __init__(
        self,
        device: torch.device | None = None,
        data_root: Path | None = None,
        **config: Any,
    ) -> None:
        super().__init__(**config)
        self.device = device or torch.device("cpu")
        self.data_root = data_root
        self._encoder: torch.nn.Module | None = None

    def _load(self) -> torch.nn.Module:
        if self._encoder is None:
            if self.data_root is None:
                from noisemismatch.config import load_paths

                self.data_root = load_paths().data_root
            c = self.config
            path = checkpoint_dir(self.data_root, c["repo"], c["revision"]) / c["file"]
            if sha256(path) != c["sha256"]:
                raise ValueError(f"{path}: SHA-256 differs from the configuration")
            self._encoder = load_encoder(str(path), self.device)
        return self._encoder

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """Embedding of one waveform, shape ``(6912,)``."""
        return self.batch(waveform[None])[0]

    def batch(self, waveforms: np.ndarray) -> np.ndarray:
        """Embeddings of several waveforms, shape ``(n_clips, 6912)``."""
        disable_tf32()
        encoder = self._load()
        c = self.config
        out = []
        with torch.no_grad():
            for start in range(0, len(waveforms), int(c["batch_size"])):
                block = torch.from_numpy(
                    np.ascontiguousarray(
                        waveforms[start : start + int(c["batch_size"])]
                    )
                ).to(self.device, dtype=torch.float32)
                x = filterbank(
                    block,
                    int(c["num_mel_bins"]),
                    float(c["frame_shift_ms"]),
                    int(c["target_length"]),
                    float(c["norm_mean"]),
                    float(c["norm_std"]),
                )
                out.append(pool(encoder(x)).cpu().numpy())
        return np.concatenate(out).astype(np.float32)
