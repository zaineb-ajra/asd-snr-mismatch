"""MobileNetV2 machine-ID classifier (ID-classification family).

Port of the DCASE 2022 Task 2 MobileNetV2 baseline
(https://github.com/Kota-Dohi/dcase2022_task2_baseline_mobile_net_v2, commit
aaf3932) from Keras 2.8 to PyTorch: features, windowing, training setup and
anomaly score. MIT License, Copyright (c) 2022 Kota-Dohi; its terms, which apply
to the ported parts:

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

The network (``KerasMobileNetV2``, ``InvertedResidual``, ``make_divisible``,
``SamePad``) is a PyTorch translation of ``keras.applications.mobilenet_v2``
(Keras 2.8.0), Copyright 2018 The TensorFlow Authors, licensed under the Apache
License, Version 2.0 (``LICENSES/Apache-2.0.txt``). Changes: rewritten in
PyTorch (layers, TensorFlow "same" padding as an explicit module); only the
configuration used here is kept (``alpha=0.5``, no pretrained weights, top =
global average pooling + dense layer). The network follows it with ``alpha=0.5``,
no pretrained weights and its classification top.

- input: 64 consecutive frames of a 128-band log-mel spectrogram, an image of
  64 x 128 replicated on 3 channels; training uses every 8th window, scoring all;
- task: classify the machine ID of the clip within its machine type (DCASE
  "sections"); one model per machine type;
- training: categorical cross-entropy, Adam (lr 1e-5, epsilon 1e-7), 20 epochs,
  batch 32, shuffled;
- anomaly score of a clip: mean over its windows of ``log(1 - p) - log(p)``, with
  ``p`` the probability of the clip's own machine ID (both terms floored at the
  machine epsilon).

Keras details reproduced: channel widths rounded by ``_make_divisible``, TensorFlow
"same" (asymmetric) padding of stride-2 convolutions, batch normalization with
epsilon 1e-3 and moving-average momentum 0.999, ReLU6, Glorot-uniform kernels and
zero biases. Deviations: docs/deviations.md, "MobileNetV2".
"""

from __future__ import annotations

import logging
import math
import sys
from typing import Any, ClassVar

import numpy as np
import torch
from torch import nn

from noisemismatch.data.dataset import ClipSet
from noisemismatch.features.logmel import LogMel
from noisemismatch.models.ae import keras_validation_split
from noisemismatch.models.base import (
    AnomalyDetector,
    batch_indices,
    register,
    training_steps,
)

logger = logging.getLogger(__name__)

EPS: float = sys.float_info.epsilon
# (filters, stride, expansion) of the 17 inverted residual blocks of MobileNetV2.
BLOCKS: tuple[tuple[int, int, int], ...] = (
    (16, 1, 1),
    (24, 2, 6),
    (24, 1, 6),
    (32, 2, 6),
    (32, 1, 6),
    (32, 1, 6),
    (64, 2, 6),
    (64, 1, 6),
    (64, 1, 6),
    (64, 1, 6),
    (96, 1, 6),
    (96, 1, 6),
    (96, 1, 6),
    (160, 2, 6),
    (160, 1, 6),
    (160, 1, 6),
    (320, 1, 6),
)


def make_divisible(value: float, divisor: int = 8) -> int:
    """Channel rounding of ``keras.applications.mobilenet_v2._make_divisible``."""
    rounded = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if rounded < 0.9 * value:
        rounded += divisor
    return rounded


class SamePad(nn.Module):
    """TensorFlow "same" padding, asymmetric for even sizes and stride 2.

    Args:
        kernel: Kernel size.
        stride: Stride.
    """

    def __init__(self, kernel: int, stride: int) -> None:
        super().__init__()
        self.kernel = kernel
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pads the last two dimensions."""
        pads = []
        for size in (x.shape[-1], x.shape[-2]):  # width then height (F.pad order)
            out = math.ceil(size / self.stride)
            total = max((out - 1) * self.stride + self.kernel - size, 0)
            pads += [total // 2, total - total // 2]
        return nn.functional.pad(x, pads)


def _conv_bn(
    in_ch: int,
    out_ch: int,
    kernel: int,
    stride: int,
    groups: int,
    relu6: bool,
    bn_eps: float,
    bn_momentum: float,
) -> nn.Sequential:
    """Convolution (TF "same" padding, no bias) + batch norm (+ ReLU6)."""
    layers: list[nn.Module] = [
        SamePad(kernel, stride),
        nn.Conv2d(in_ch, out_ch, kernel, stride, groups=groups, bias=False),
        nn.BatchNorm2d(out_ch, eps=bn_eps, momentum=bn_momentum),
    ]
    if relu6:
        layers.append(nn.ReLU6())
    return nn.Sequential(*layers)


class InvertedResidual(nn.Module):
    """Inverted residual block of Keras MobileNetV2 (``_inverted_res_block``)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        expansion: int,
        bn_eps: float,
        bn_momentum: float,
    ) -> None:
        super().__init__()
        hidden = in_ch * expansion
        layers: list[nn.Module] = []
        if expansion != 1:  # the first block has no expansion convolution
            layers.append(_conv_bn(in_ch, hidden, 1, 1, 1, True, bn_eps, bn_momentum))
        layers.append(
            _conv_bn(hidden, hidden, 3, stride, hidden, True, bn_eps, bn_momentum)
        )
        layers.append(_conv_bn(hidden, out_ch, 1, 1, 1, False, bn_eps, bn_momentum))
        self.block = nn.Sequential(*layers)
        self.residual = in_ch == out_ch and stride == 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the block (with the identity shortcut when shapes match)."""
        y = self.block(x)
        return x + y if self.residual else y


class KerasMobileNetV2(nn.Module):
    """MobileNetV2 with the Keras layout, width multiplier and classification top.

    Args:
        n_classes: Number of output classes.
        alpha: Width multiplier.
        bn_eps: Batch-norm epsilon.
        bn_momentum: Batch-norm momentum (PyTorch convention: 1 - Keras momentum).
    """

    def __init__(
        self,
        n_classes: int,
        alpha: float = 0.5,
        bn_eps: float = 1e-3,
        bn_momentum: float = 1e-3,
    ) -> None:
        super().__init__()
        channels = make_divisible(32 * alpha)
        layers: list[nn.Module] = [
            _conv_bn(3, channels, 3, 2, 1, True, bn_eps, bn_momentum)
        ]
        for filters, stride, expansion in BLOCKS:
            out_ch = make_divisible(int(filters * alpha))
            layers.append(
                InvertedResidual(
                    channels, out_ch, stride, expansion, bn_eps, bn_momentum
                )
            )
            channels = out_ch
        last = 1280 if alpha <= 1.0 else make_divisible(1280 * alpha)
        layers.append(_conv_bn(channels, last, 1, 1, 1, True, bn_eps, bn_momentum))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(last, n_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d | nn.Linear):
                nn.init.xavier_uniform_(module.weight)  # Keras glorot_uniform
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Class logits of images shaped ``(batch, 1, frames, mels)``."""
        x = self.features(x.expand(-1, 3, -1, -1))  # grey image on 3 channels
        return self.classifier(x.mean(dim=(2, 3)))


def window_images(features: np.ndarray, n_frames: int, hop: int) -> np.ndarray:
    """Cuts log-mel spectrograms into images of ``n_frames`` consecutive frames.

    Args:
        features: Array shaped ``(n_clips, n_mels, n_total_frames)``.
        n_frames: Frames per image.
        hop: Keep one window out of ``hop`` (1 = all windows).

    Returns:
        Array shaped ``(n_clips, n_windows, n_frames, n_mels)``.
    """
    windows = np.lib.stride_tricks.sliding_window_view(features, n_frames, axis=2)
    # windows[c, m, s, t] = features[c, m, s + t]  ->  (c, s, t, m)
    return windows[:, :, ::hop, :].transpose(0, 2, 3, 1)


@register
class MobileNetV2Detector(AnomalyDetector):
    """DCASE 2022 MobileNetV2 baseline (see module docstring)."""

    name: ClassVar[str] = "mobilenetv2"
    training_scope: ClassVar[str] = "machine_type"

    def __init__(self, config: dict[str, Any], seed: int, device: torch.device) -> None:
        super().__init__(config, seed, device)
        self.extractor = LogMel(**config["features"])
        self.network: KerasMobileNetV2 | None = None
        self.classes: list[str] = []
        self.history: list[dict[str, float]] = []
        self.steps = 0
        self.n_train_images = 0
        self.n_val_images = 0

    def _images(self, clips: ClipSet, hop: int) -> np.ndarray:
        features = clips.features(self.extractor)
        return window_images(features, int(self.config["n_frames"]), hop)

    def _labels(self, clips: ClipSet) -> np.ndarray:
        index = {name: k for k, name in enumerate(self.classes)}
        return np.array([index[m] for m in clips.table["machine_id"]])

    def fit(self, train_set: ClipSet) -> None:
        """Trains the machine-ID classifier on the normal training clips.

        Images are ordered by clip (identity, then SNR) and window; the Keras
        validation split, if any, holds out the last images. Multi-condition
        runs get the step budget of a single-SNR run.

        Args:
            train_set: Normal training clips of one machine type.
        """
        c = self.config
        self.classes = sorted(train_set.table["machine_id"].unique())
        images = self._images(train_set, int(c["n_hop_frames"]))
        n_windows = images.shape[1]
        x = np.ascontiguousarray(
            images.reshape(-1, *images.shape[2:]), dtype=np.float32
        )
        y = np.repeat(self._labels(train_set), n_windows)
        fraction = float(c["validation_split"])
        split_at = keras_validation_split(len(x), fraction)
        n_snrs = int(train_set.table["snr_db"].nunique())
        single = keras_validation_split(len(x) // n_snrs, fraction)
        batch = int(c["batch_size"])
        steps = training_steps(single, batch, int(c["epochs"]), drop_last=False)
        steps_per_epoch = steps // int(c["epochs"])
        self.steps, self.n_train_images, self.n_val_images = (
            steps,
            split_at,
            len(x) - split_at,
        )
        # Recorded with the run (step budget of the multi-condition model).
        self.diagnostics = {"steps": steps, "n_train_images": split_at}

        self.network = KerasMobileNetV2(
            len(self.classes),
            float(c["alpha"]),
            float(c["bn_eps"]),
            float(c["bn_momentum"]),
        ).to(self.device)
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=float(c["learning_rate"]),
            betas=tuple(c["adam_betas"]),
            eps=float(c["adam_eps"]),
        )
        rng = np.random.default_rng(self.seed)
        loss_sum = correct = count = 0.0
        self.network.train()
        for step, idx in enumerate(batch_indices(split_at, batch, steps, rng), 1):
            idx = np.sort(idx)  # faster gather; batch content is unchanged
            xb = (
                torch.from_numpy(x[idx]).unsqueeze(1).to(self.device, non_blocking=True)
            )
            yb = torch.from_numpy(y[idx]).to(self.device)
            logits = self.network(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(idx)
            correct += float((logits.argmax(1) == yb).sum())
            count += len(idx)
            if step % steps_per_epoch == 0:
                record = {
                    "epoch": step // steps_per_epoch,
                    "loss": loss_sum / count,
                    "accuracy": correct / count,
                }
                if split_at < len(x):
                    record.update(self._evaluate(x[split_at:], y[split_at:]))
                self.history.append(record)
                logger.info("MobileNetV2 %s", record)
                loss_sum = correct = count = 0.0

    def _evaluate(
        self, x: np.ndarray, y: np.ndarray, batch: int = 1024
    ) -> dict[str, float]:
        """Validation loss and accuracy (network switched back to training mode)."""
        assert self.network is not None
        self.network.eval()
        loss = correct = 0.0
        with torch.no_grad():
            for start in range(0, len(x), batch):
                xb = (
                    torch.from_numpy(x[start : start + batch])
                    .unsqueeze(1)
                    .to(self.device)
                )
                yb = torch.from_numpy(y[start : start + batch]).to(self.device)
                logits = self.network(xb)
                loss += float(nn.functional.cross_entropy(logits, yb, reduction="sum"))
                correct += float((logits.argmax(1) == yb).sum())
        self.network.train()
        return {"val_loss": loss / len(x), "val_accuracy": correct / len(x)}

    def id_probabilities(self, clips: ClipSet, chunk: int = 16) -> np.ndarray:
        """Machine-ID probabilities of every window, per clip.

        Args:
            clips: Clips of the machine type the model was trained on.
            chunk: Clips processed at once.

        Returns:
            Array shaped ``(n_clips, n_windows, n_classes)``.
        """
        if self.network is None:
            raise RuntimeError("fit must be called before scoring")
        images = self._images(clips, hop=1)
        out = []
        self.network.eval()
        with torch.no_grad():
            for start in range(0, len(images), chunk):
                block = np.ascontiguousarray(
                    images[start : start + chunk], dtype=np.float32
                )
                xb = torch.from_numpy(block.reshape(-1, 1, *block.shape[2:])).to(
                    self.device
                )
                p = torch.softmax(self.network(xb), dim=1).cpu().numpy()
                out.append(p.reshape(len(block), block.shape[1], -1))
        return np.concatenate(out)

    def score(self, test_set: ClipSet) -> np.ndarray:
        """Anomaly scores: mean over windows of ``log(1 - p) - log(p)``.

        Args:
            test_set: Test clips of the machine type.

        Returns:
            Float array shaped ``(len(test_set),)``.
        """
        probabilities = self.id_probabilities(test_set)
        own = self._labels(test_set)
        p = probabilities[np.arange(len(own)), :, own]  # (n_clips, n_windows)
        scores = np.log(np.maximum(1.0 - p, EPS)) - np.log(np.maximum(p, EPS))
        return scores.mean(axis=1).astype(float)
