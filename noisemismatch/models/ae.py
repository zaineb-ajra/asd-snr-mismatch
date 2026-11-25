"""Dense autoencoder of the MIMII baseline (reconstruction family).

Re-implemented in PyTorch from the MIMII paper (Sec. 4) and the hyperparameters of
the baseline code (``baseline.py`` / ``baseline.yaml``, MIMII-hitachi/mimii_baseline,
commit 7086d21; no code copied, the repository has no license):

- input: 5 consecutive frames of a 64-band log-mel spectrogram (320 values);
- network: dense layers 64-64-8-64-64 with ReLU, linear 320-unit output;
- training: MSE, Adam, 50 epochs, batch 512, shuffled; Keras ``validation_split``
  supported but set to 0.0 in the study (docs/deviations.md, A6);
- score of a clip: mean over its vectors of the mean squared reconstruction error.

Framework defaults that the baseline code relies on without stating them (Keras
2.1.6) are reproduced: Glorot-uniform weights and zero biases, Adam with learning
rate 0.001, betas (0.9, 0.999) and epsilon 1e-7. One model is trained per machine
type and machine ID.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np
import torch
from torch import nn

from noisemismatch.data.dataset import ClipSet
from noisemismatch.features.logmel import LogMel
from noisemismatch.models.base import (
    AnomalyDetector,
    batch_indices,
    register,
    training_steps,
)

logger = logging.getLogger(__name__)


def frame_vectors(features: np.ndarray, frames: int) -> np.ndarray:
    """Concatenates ``frames`` consecutive frames into vectors (baseline layout).

    Vector ``s`` of a clip is ``[f(s), f(s+1), ..., f(s+frames-1)]`` where ``f(t)``
    is the log-mel frame ``t`` (all bands), as in the baseline.

    Args:
        features: Log-mel arrays shaped ``(n_clips, n_mels, n_frames)``.
        frames: Number of consecutive frames per vector.

    Returns:
        Array shaped ``(n_clips, n_frames - frames + 1, frames * n_mels)``.
    """
    n_clips, _, n_frames = features.shape
    windows = np.lib.stride_tricks.sliding_window_view(features, frames, axis=2)
    # windows[c, m, s, t] = features[c, m, s + t]  ->  (c, s, t, m)
    vectors = windows.transpose(0, 2, 3, 1)
    return np.ascontiguousarray(vectors).reshape(n_clips, n_frames - frames + 1, -1)


def keras_validation_split(n: int, fraction: float) -> int:
    """Index where Keras splits training from validation samples.

    Keras keeps ``x[:split_at]`` for training and ``x[split_at:]`` (the last
    samples, before any shuffling) for validation.

    Args:
        n: Number of samples.
        fraction: ``validation_split``.

    Returns:
        ``split_at``.
    """
    return int(n * (1.0 - fraction))


class _Network(nn.Module):
    """Dense autoencoder with ReLU hidden layers and a linear output."""

    def __init__(self, dim: int, hidden: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = dim
        for units in hidden:
            layers += [nn.Linear(previous, units), nn.ReLU()]
            previous = units
        layers.append(nn.Linear(previous, dim))
        self.layers = nn.Sequential(*layers)
        for module in self.layers:
            if isinstance(module, nn.Linear):  # Keras Dense defaults
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstructs the input vectors."""
        return self.layers(x)


@register
class MimiiAutoencoder(AnomalyDetector):
    """MIMII baseline autoencoder (see module docstring)."""

    name: ClassVar[str] = "ae"
    training_scope: ClassVar[str] = "machine_id"

    def __init__(self, config: dict[str, Any], seed: int, device: torch.device) -> None:
        super().__init__(config, seed, device)
        self.extractor = LogMel(**config["features"])
        self.frames = int(config["frames"])
        self.network: _Network | None = None
        self.history: list[dict[str, float]] = []
        self.n_train_vectors = 0
        self.n_val_vectors = 0
        self.steps = 0

    def _vectors(self, clips: ClipSet) -> np.ndarray:
        """Vectors of every clip, shaped ``(n_clips, n_vectors, dim)``."""
        return frame_vectors(clips.features(self.extractor), self.frames)

    def fit(self, train_set: ClipSet) -> None:
        """Trains the autoencoder on the vectors of the normal training clips.

        Vectors are concatenated in clip order (identity, then SNR), the Keras
        validation split is applied to that sequence, and the number of
        optimization steps is that of a single-SNR run (multi-condition runs get
        the same number of steps on the union).

        Args:
            train_set: Normal training clips of one machine ID.
        """
        c = self.config
        vectors = self._vectors(train_set)
        x = vectors.reshape(-1, vectors.shape[-1])
        fraction = float(c["validation_split"])
        split_at = keras_validation_split(len(x), fraction)
        x_train, x_val = x[:split_at], x[split_at:]
        n_snrs = int(train_set.table["snr_db"].nunique())
        single = keras_validation_split(len(x) // n_snrs, fraction)
        batch = int(c["batch_size"])
        steps = training_steps(single, batch, int(c["epochs"]), drop_last=False)
        steps_per_epoch = steps // int(c["epochs"])
        self.n_train_vectors, self.n_val_vectors, self.steps = (
            len(x_train),
            len(x_val),
            steps,
        )
        # Recorded with the run (step budget of the multi-condition model).
        self.diagnostics = {"steps": steps, "n_train_vectors": len(x_train)}

        self.network = _Network(x.shape[1], list(c["hidden_units"])).to(self.device)
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=float(c["learning_rate"]),
            betas=tuple(c["adam_betas"]),
            eps=float(c["adam_eps"]),
        )
        train_t = torch.from_numpy(x_train).to(self.device)
        val_t = torch.from_numpy(x_val).to(self.device) if len(x_val) else None
        rng = np.random.default_rng(self.seed)
        running, count = 0.0, 0
        self.network.train()
        for step, idx in enumerate(batch_indices(len(x_train), batch, steps, rng), 1):
            xb = train_t[torch.from_numpy(idx).to(self.device)]
            loss = nn.functional.mse_loss(self.network(xb), xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(idx)
            count += len(idx)
            if step % steps_per_epoch == 0:
                record = {"epoch": step // steps_per_epoch, "loss": running / count}
                if val_t is not None:
                    record["val_loss"] = self._mse(val_t)
                self.history.append(record)
                running, count = 0.0, 0
        last = self.history[-1] if self.history else {}
        logger.info(
            "AE: %d training vectors (%d validation), %d steps, final %s",
            len(x_train),
            len(x_val),
            steps,
            last,
        )

    def _mse(self, x: torch.Tensor, batch: int = 8192) -> float:
        """Mean squared reconstruction error over ``x``."""
        assert self.network is not None
        self.network.eval()
        total = 0.0
        with torch.no_grad():
            for start in range(0, len(x), batch):
                xb = x[start : start + batch]
                total += float(((self.network(xb) - xb) ** 2).mean(dim=1).sum())
        self.network.train()
        return total / len(x)

    def score(self, test_set: ClipSet) -> np.ndarray:
        """Mean reconstruction error of each clip (higher = more anomalous).

        Args:
            test_set: Test clips.

        Returns:
            Float array shaped ``(len(test_set),)``.
        """
        if self.network is None:
            raise RuntimeError("fit must be called before score")
        vectors = torch.from_numpy(self._vectors(test_set)).to(self.device)
        self.network.eval()
        with torch.no_grad():
            error = ((self.network(vectors) - vectors) ** 2).mean(dim=2)  # per vector
        return error.mean(dim=1).cpu().numpy().astype(float)
