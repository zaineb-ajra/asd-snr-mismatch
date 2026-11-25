"""Wilkinghoff SSL / FeatEx model (angular-margin family).

Re-implemented from the author's articles only (docs/deviations.md, "Angular
margin with self-supervision"): the "proposed approach" of Wilkinghoff, ICASSP
2024 (arXiv 2312.09578), with the architecture of his ICASSP 2023 paper and
thesis (Tables 14-15), the sub-cluster AdaCos loss of IJCNN 2021 / TASLP 2024
(Definition 4), the StatEx variant and FeatEx of the thesis (Definitions 5.5,
5.6) and the backend of the ICASSP 2024 paper. No code of wilkinghoff/ssl4asd is
used.

- inputs: magnitude STFT (Hann 1024, hop 512, no padding, temporal mean
  normalization) and magnitude DFT of the whole waveform (first N/2 bins);
- network: a ResNet-style branch for the spectrogram and a 1D-convolutional
  branch for the spectrum, each ending in a 128-d embedding (no bias anywhere,
  batch normalization and ReLU after every hidden layer), concatenated (256);
- training: two sub-cluster AdaCos losses (16 sub-clusters per class): the
  meta-information classes with mixed-up labels and fixed random centers, and
  the 9N classes of the sequentially applied StatEx variant and FeatEx with
  trainable centers; mixup on waveforms; 10 epochs, batch 64, Adam;
- backend: per group (machine type and ID, or DCASE section), k-means with 16
  clusters on the source-domain training embeddings, plus the target-domain
  training embeddings if any; anomaly score = smallest cosine distance.
"""

from __future__ import annotations

import logging
import math
from typing import Any, ClassVar

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn

from noisemismatch.data.dataset import ClipSet
from noisemismatch.models.base import (
    AnomalyDetector,
    batch_indices,
    register,
    training_steps,
)

logger = logging.getLogger(__name__)

SOURCE: str = "source"
TARGET: str = "target"


def same_padding(size: int, kernel: int, stride: int) -> tuple[int, int]:
    """TensorFlow "same" padding of one dimension: (before, after)."""
    out = math.ceil(size / stride)
    total = max((out - 1) * stride + kernel - size, 0)
    return total // 2, total - total // 2


class ConvBN(nn.Module):
    """Convolution without bias ("same" padding), batch norm, optional ReLU.

    Args:
        dims: 1 or 2.
        in_ch: Input channels.
        out_ch: Output channels.
        kernel: Kernel size.
        stride: Stride.
        relu: Apply ReLU.
        bn: Batch-norm keyword arguments (``eps``, ``momentum``).
    """

    def __init__(
        self,
        dims: int,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int,
        relu: bool,
        bn: dict[str, float],
    ) -> None:
        super().__init__()
        conv = nn.Conv1d if dims == 1 else nn.Conv2d
        norm = nn.BatchNorm1d if dims == 1 else nn.BatchNorm2d
        self.dims, self.kernel, self.stride = dims, kernel, stride
        self.conv = conv(in_ch, out_ch, kernel, stride=stride, bias=False)
        self.norm = norm(out_ch, **bn)
        self.relu = relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pads, convolves, normalizes (and applies ReLU)."""
        pads: list[int] = []
        for size in reversed(x.shape[2:]):  # last dimension first (F.pad order)
            pads += same_padding(int(size), self.kernel, self.stride)
        x = self.norm(self.conv(nn.functional.pad(x, pads)))
        return torch.relu(x) if self.relu else x


class BasicBlock(nn.Module):
    """ResNet basic block (He et al. 2016): two 3x3 convolutions, residual add.

    A 1x1 convolution (with batch norm) projects the shortcut when the stride or
    the number of channels changes.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int, bn: dict) -> None:
        super().__init__()
        self.first = ConvBN(2, in_ch, out_ch, 3, stride, True, bn)
        self.second = ConvBN(2, out_ch, out_ch, 3, 1, False, bn)
        self.shortcut = (
            ConvBN(2, in_ch, out_ch, 1, stride, False, bn)
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual block output."""
        return torch.relu(self.second(self.first(x)) + self.shortcut(x))


class SpectrogramNet(nn.Module):
    """Spectrogram branch (thesis Table 15): stem, four stages, time max-pooling.

    Args:
        n_freq: Frequency bins of the input (513).
        dim: Embedding size.
        bn: Batch-norm keyword arguments.
    """

    STAGES: ClassVar[tuple[tuple[int, int], ...]] = (
        (16, 1),
        (32, 2),
        (64, 2),
        (128, 2),
    )

    def __init__(self, n_freq: int, dim: int, bn: dict) -> None:
        super().__init__()
        self.stem = ConvBN(2, 1, 16, 7, 2, True, bn)
        self.pool = nn.MaxPool2d(3, stride=2)
        blocks: list[nn.Module] = []
        in_ch = 16
        for channels, stride in self.STAGES:
            blocks += [
                BasicBlock(in_ch, channels, stride, bn),
                BasicBlock(channels, channels, 1, bn),
            ]
            in_ch = channels
        self.blocks = nn.Sequential(*blocks)
        freq = (math.ceil(n_freq / 2) - 3) // 2 + 1
        for _, stride in self.STAGES[1:]:
            freq = math.ceil(freq / stride)
        self.flat_dim = freq * in_ch
        self.embedding = nn.Linear(self.flat_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 1, T, F)`` -> ``(B, dim)``."""
        x = self.blocks(self.pool(self.stem(x)))  # (B, C, T', F')
        x = x.amax(dim=2)  # max pooling over the whole time axis
        return self.embedding(x.flatten(1))


class SpectrumNet(nn.Module):
    """Spectrum branch (thesis Table 14): three 1D convolutions, five dense layers.

    Args:
        length: Input length (DFT bins).
        kernels: Kernel sizes of the three convolutions.
        strides: Strides of the three convolutions.
        channels: Channels of the convolutions.
        dim: Width of the dense layers and of the embedding.
        bn: Batch-norm keyword arguments.
    """

    def __init__(
        self,
        length: int,
        kernels: list[int],
        strides: list[int],
        channels: int,
        dim: int,
        bn: dict,
    ) -> None:
        super().__init__()
        convs, in_ch = [], 1
        for kernel, stride in zip(kernels, strides, strict=True):
            convs.append(ConvBN(1, in_ch, channels, kernel, stride, True, bn))
            in_ch = channels
            length = math.ceil(length / stride)
        self.convs = nn.Sequential(*convs)
        self.flat_dim = length * channels
        dense: list[nn.Module] = []
        width = self.flat_dim
        for _ in range(4):
            dense += [
                nn.Linear(width, dim, bias=False),
                nn.BatchNorm1d(dim, **bn),
                nn.ReLU(),
            ]
            width = dim
        self.dense = nn.Sequential(*dense)
        self.embedding = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, L)`` -> ``(B, dim)``."""
        x = self.convs(x.unsqueeze(1))
        return self.embedding(self.dense(x.flatten(1)))


class SubClusterAdaCos(nn.Module):
    """Sub-cluster AdaCos loss (IJCNN 2021; TASLP 2024, Definition 4).

    With soft (mixed-up) labels, the median angle is taken over the mixed-up
    angles ``sum_j y_j * theta_j``; the angle of a sample to a class is the angle
    to the closest sub-cluster of that class.

    Args:
        n_classes: Number of classes N.
        n_sub: Sub-clusters per class M.
        dim: Embedding size.
        trainable: Whether the centers are trained.
    """

    def __init__(self, n_classes: int, n_sub: int, dim: int, trainable: bool) -> None:
        super().__init__()
        self.n_classes, self.n_sub = n_classes, n_sub
        centers = torch.empty(dim, n_classes * n_sub)
        nn.init.xavier_uniform_(centers)  # Glorot uniform (Keras default)
        self.centers = nn.Parameter(centers, requires_grad=trainable)
        self.register_buffer(
            "scale", torch.tensor(math.sqrt(2.0) * math.log(n_classes * n_sub - 1))
        )

    def cosines(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Cosine similarities to every sub-cluster, ``(B, N * M)``."""
        e = nn.functional.normalize(embeddings, dim=1)
        c = nn.functional.normalize(self.centers, dim=0)
        return e @ c

    @torch.no_grad()
    def update_scale(self, cos: torch.Tensor, labels: torch.Tensor) -> None:
        """Adaptive scale of Definition 4 (uses the previous scale)."""
        logits = self.scale * cos
        f_max = logits.max()
        b_avg = torch.exp(logits - f_max).sum(dim=1).mean()
        per_class = cos.view(len(cos), self.n_classes, self.n_sub).amax(dim=2)
        angles = torch.arccos(per_class.clamp(-1.0, 1.0))
        theta_med = torch.median((labels * angles).sum(dim=1))
        denom = torch.cos(torch.clamp(theta_med, max=math.pi / 4))
        self.scale.copy_((f_max + torch.log(b_avg)) / denom)

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor, update: bool = True
    ) -> torch.Tensor:
        """Mean cross-entropy of the soft labels ``(B, N)``."""
        cos = self.cosines(embeddings)
        if update:
            self.update_scale(cos.detach(), labels)
        logits = self.scale * cos
        log_all = torch.logsumexp(logits, dim=1, keepdim=True)
        per_class = logits.view(len(logits), self.n_classes, self.n_sub)
        log_prob = torch.logsumexp(per_class, dim=2) - log_all
        return -(labels * log_prob).sum(dim=1).mean()


def spectrogram(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    """Magnitude STFT without padding, temporal mean removed: ``(B, 1, T, F)``."""
    window = torch.hann_window(n_fft, periodic=True, device=x.device)
    s = torch.stft(
        x, n_fft, hop, window=window, center=False, return_complex=True
    ).abs()
    s = s.transpose(1, 2)  # (B, T, F)
    return (s - s.mean(dim=1, keepdim=True)).unsqueeze(1)


def spectrum(x: torch.Tensor) -> torch.Tensor:
    """Magnitude DFT of the whole waveform, first N/2 bins: ``(B, N/2)``."""
    return torch.fft.rfft(x).abs()[:, : x.shape[1] // 2]


def statex(spec: torch.Tensor, perm: torch.Tensor, eps: float) -> torch.Tensor:
    """Temporal StatEx: exchange of the statistics over the frequency axis.

    Args:
        spec: Features ``(B, 1, T, F)``.
        perm: Partner of each sample.
        eps: Added to the standard deviation.

    Returns:
        ``(x1 - mu1) / sigma1 * sigma2 + mu2``, statistics per time frame.
    """
    mu = spec.mean(dim=3, keepdim=True)
    sigma = spec.std(dim=3, keepdim=True) + eps
    return (spec - mu) / sigma * sigma[perm] + mu[perm]


def exchange_labels(
    y: torch.Tensor, perm: torch.Tensor, apply: torch.Tensor
) -> torch.Tensor:
    """(0, y1/2, y2/2) where applied, (y, 0, 0) elsewhere; tripled class count."""
    zeros = torch.zeros_like(y)
    new = torch.cat([zeros, 0.5 * y, 0.5 * y[perm]], dim=1)
    old = torch.cat([y, zeros, zeros], dim=1)
    return torch.where(apply[:, None], new, old)


def adjust_length(
    wave: np.ndarray, length: int, rng: np.random.Generator
) -> np.ndarray:
    """Repeats a waveform to at least ``length`` samples and crops it at random."""
    if len(wave) == length:
        return wave
    reps = math.ceil(length / len(wave))
    tiled = np.tile(wave, reps)
    start = int(rng.integers(len(tiled) - length + 1))
    return tiled[start : start + length]


class WilkinghoffNet(nn.Module):
    """Both branches and their concatenated embedding."""

    def __init__(self, length: int, c: dict[str, Any]) -> None:
        super().__init__()
        bn = {"eps": float(c["bn_eps"]), "momentum": float(c["bn_momentum"])}
        n_frames = (length - int(c["n_fft"])) // int(c["hop"]) + 1
        n_freq = int(c["n_fft"]) // 2 + 1
        self.n_fft, self.hop, self.n_frames = int(c["n_fft"]), int(c["hop"]), n_frames
        self.spec_net = SpectrogramNet(n_freq, int(c["embedding_dim"]), bn)
        self.dft_net = SpectrumNet(
            length // 2,
            list(c["spectrum_kernels"]),
            list(c["spectrum_strides"]),
            int(c["spectrum_channels"]),
            int(c["embedding_dim"]),
            bn,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d | nn.Conv2d | nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def branches(
        self, spec: torch.Tensor, dft: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embeddings of the spectrogram and spectrum branches."""
        return self.spec_net(spec), self.dft_net(dft)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenated embedding of waveforms ``(B, L)``."""
        e1, e2 = self.branches(spectrogram(x, self.n_fft, self.hop), spectrum(x))
        return torch.cat([e1, e2], dim=1)


@register
class WilkinghoffDetector(AnomalyDetector):
    """Wilkinghoff "proposed approach" (see module docstring).

    Classes are the ``class_label`` column of the training table if present
    (DCASE: machine type, section and attributes), else machine type x machine
    ID. The backend groups clips by (machine type, machine ID); a ``domain``
    column, if present, separates source and target training clips.
    """

    name: ClassVar[str] = "wilkinghoff"
    training_scope: ClassVar[str] = "all"

    def __init__(self, config: dict[str, Any], seed: int, device: torch.device) -> None:
        super().__init__(config, seed, device)
        self.net: WilkinghoffNet | None = None
        self.classes: list[str] = []
        self.backend: dict[tuple[str, str], torch.Tensor] = {}
        self.history: list[dict[str, float]] = []
        self.steps = 0
        # Classes without training clips, added to the class list (fidelity
        # check on DCASE only; docs/deviations.md).
        self.extra_classes: list[str] = []

    @staticmethod
    def class_labels(table: Any) -> np.ndarray:
        """Class of each clip."""
        if "class_label" in table:
            return table["class_label"].astype(str).to_numpy()
        return (table["machine_type"] + "/" + table["machine_id"]).to_numpy()

    def _waveforms(self, clips: ClipSet, rng: np.random.Generator) -> np.ndarray:
        raw = clips.raw_waveforms()
        length = int(self.config["input_length"])
        if raw.shape[1] == length or hasattr(clips.audio, "native_lengths"):
            return raw  # extended storage: cropped per step (training) or at 0
        return np.stack([adjust_length(w, length, rng) for w in raw])

    def _crop_max(self, clips: ClipSet, stored: int) -> np.ndarray:
        """Largest crop start of each clip in its repeated waveform.

        A clip of n samples repeated to the first multiple of n that reaches the
        input length L can be cropped at any start in [0, ceil(L / n) * n - L].
        """
        length = int(self.config["input_length"])
        if stored == length:
            return np.zeros(len(clips), dtype=np.int64)
        native = clips.gather(clips.audio.native_lengths, np.int64)
        return np.minimum(
            np.ceil(length / native).astype(np.int64) * native - length, stored - length
        )

    def fit(self, train_set: ClipSet) -> None:
        """Trains the embedding model, then builds the backend.

        Args:
            train_set: Normal training clips of every machine type.
        """
        c = self.config
        rng = np.random.default_rng(self.seed)
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        labels = self.class_labels(train_set.table)
        self.classes = sorted(set(labels) | set(self.extra_classes))
        index = {name: k for k, name in enumerate(self.classes)}
        y_all = torch.tensor([index[v] for v in labels])
        raw = self._waveforms(train_set, rng)
        length = int(c["input_length"])
        crop_max = self._crop_max(train_set, raw.shape[1])
        crop_rng = np.random.default_rng([self.seed, 1])
        n_classes = len(self.classes)
        self.net = WilkinghoffNet(length, c).to(self.device)
        dim = 2 * int(c["embedding_dim"])
        self.loss_main = SubClusterAdaCos(
            n_classes, int(c["n_subclusters"]), dim, False
        ).to(self.device)
        self.loss_ssl = SubClusterAdaCos(
            9 * n_classes, int(c["n_subclusters"]), dim, True
        ).to(self.device)
        params = [*self.net.parameters(), self.loss_ssl.centers]
        optimizer = torch.optim.Adam(
            params,
            lr=float(c["learning_rate"]),
            betas=tuple(c["adam_betas"]),
            eps=float(c["adam_eps"]),
        )
        batch = int(c["batch_size"])
        n_snrs = int(train_set.table["snr_db"].nunique())
        steps = training_steps(
            len(raw) // n_snrs, batch, int(c["epochs"]), drop_last=False
        )
        steps_per_epoch = max(steps // int(c["epochs"]), 1)
        self.steps = steps
        self.net.train()
        running = 0.0
        for step, idx in enumerate(batch_indices(len(raw), batch, steps, rng), 1):
            if raw.shape[1] == length:
                batch_raw = raw[idx]
            else:  # random crop of the repeated waveform at every step
                starts = crop_rng.integers(0, crop_max[idx] + 1)
                batch_raw = np.stack(
                    [raw[i, s : s + length] for i, s in zip(idx, starts, strict=True)]
                )
            loss = self._step(batch_raw, y_all[idx], n_classes, gen)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
            if step % steps_per_epoch == 0:
                record = {
                    "epoch": step // steps_per_epoch,
                    "loss": running / steps_per_epoch,
                }
                self.history.append(record)
                logger.info("Wilkinghoff %s", record)
                running = 0.0
        self._build_backend(train_set, rng)
        # Recorded with the run (step budget of the multi-condition model).
        self.diagnostics = {
            "steps": self.steps,
            "n_train_clips": len(raw),
            "n_classes": n_classes,
        }

    def _step(
        self, raw: np.ndarray, y: torch.Tensor, n_classes: int, gen: torch.Generator
    ) -> torch.Tensor:
        c = self.config
        dev = self.device
        b = len(raw)
        x = torch.from_numpy(raw).to(dev, torch.float32) / 32768.0
        y1 = nn.functional.one_hot(y, n_classes).to(dev, torch.float32)

        def draw(prob: float) -> tuple[torch.Tensor, torch.Tensor]:
            perm = torch.randperm(b, generator=gen).to(dev)
            apply = (torch.rand(b, generator=gen) < prob).to(dev)
            return perm, apply

        perm, apply = draw(float(c["mixup_prob"]))
        lam = torch.rand(b, generator=gen).to(dev)
        lam = torch.where(apply, lam, torch.ones_like(lam))
        x = lam[:, None] * x + (1 - lam[:, None]) * x[perm]
        y_mix = lam[:, None] * y1 + (1 - lam[:, None]) * y1[perm]
        spec = spectrogram(x, self.net.n_fft, self.net.hop)
        dft = spectrum(x)
        perm, apply = draw(float(c["statex_prob"]))
        spec = torch.where(
            apply[:, None, None, None], statex(spec, perm, float(c["statex_eps"])), spec
        )
        y3 = exchange_labels(y_mix, perm, apply)
        e1, e2 = self.net.branches(spec, dft)
        perm, apply = draw(float(c["featex_prob"]))
        e_ssl = torch.cat([e1, torch.where(apply[:, None], e2[perm], e2)], dim=1)
        y9 = exchange_labels(y3, perm, apply)
        return self.loss_main(torch.cat([e1, e2], dim=1), y_mix) + self.loss_ssl(
            e_ssl, y9
        )

    def embed(self, clips: ClipSet, chunk: int = 64) -> torch.Tensor:
        """L2-normalized embeddings of clips (no augmentation), on the device."""
        if self.net is None:
            raise RuntimeError("fit must be called first")
        raw = self._waveforms(clips, np.random.default_rng(self.seed))
        raw = raw[:, : int(self.config["input_length"])]  # extended storage: start 0
        self.net.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(raw), chunk):
                x = torch.from_numpy(raw[start : start + chunk]).to(
                    self.device, torch.float32
                )
                out.append(self.net(x / 32768.0))
        return nn.functional.normalize(torch.cat(out), dim=1)

    def _build_backend(self, train_set: ClipSet, rng: np.random.Generator) -> None:
        c = self.config
        emb = self.embed(train_set)
        table = train_set.table.reset_index(drop=True)
        domains = table.get("domain")
        for key, part in table.groupby(["machine_type", "machine_id"], sort=True):
            rows = part.index.to_numpy()
            source = (
                rows if domains is None else rows[(domains[rows] != TARGET).to_numpy()]
            )
            target = (
                np.array([], dtype=int)
                if domains is None
                else rows[(domains[rows] == TARGET).to_numpy()]
            )
            k = min(int(c["kmeans_clusters"]), len(source))
            means = (
                KMeans(n_clusters=k, random_state=self.seed, n_init="auto")
                .fit(emb[source].cpu().numpy())
                .cluster_centers_
            )
            refs = [torch.from_numpy(means).to(self.device, torch.float32)]
            if len(target):
                refs.append(emb[target])
            self.backend[tuple(key)] = nn.functional.normalize(torch.cat(refs), dim=1)

    def score(self, test_set: ClipSet) -> np.ndarray:
        """Smallest cosine distance to the means and target embeddings of the group.

        Args:
            test_set: Test clips.

        Returns:
            Float array shaped ``(len(test_set),)``.
        """
        emb = self.embed(test_set)
        table = test_set.table.reset_index(drop=True)
        scores = np.empty(len(table))
        for key, part in table.groupby(["machine_type", "machine_id"], sort=True):
            rows = part.index.to_numpy()
            refs = self.backend[tuple(key)]
            scores[rows] = (1.0 - (emb[rows] @ refs.T).amax(dim=1)).cpu().numpy()
        return scores
