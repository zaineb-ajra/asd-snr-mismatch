"""OS-SCL: noise-supervised contrastive learning with feature perturbation.

Re-implemented from Huang et al., ICASSP 2025 (arXiv 2509.13853), variant
TFSTgram (OS-SCL), and the papers it cites (STgram-MFN, Liu et al. 2022;
MobileFaceNets, Chen et al. 2018; Noisy-ArcMix, Choi et al. 2024; ArcFace; mixup);
values none of them gives are documented facts of the huangswt/OS-SCL repository
(no code of it is used). The source of every element is listed in
docs/deviations.md ("OS-SCL").

- input: Concat(log-Mel spectrogram, Tgram, TFgram), 3 x 128 x 313;
- embedding: MobileFaceNet (128-d), feature perturbation head (128 -> 64 -> 128),
  L2 normalization;
- training: mixup of waveform and log-Mel spectrogram (lambda ~ Beta(0.5, 0.5));
  loss = noise-supervised contrastive loss (original labels, tau = 0.02) +
  Noisy-ArcMix (ArcFace s = 30, m = 0.4, margin on the original label);
  AdamW, cosine annealing, EMA of the weights;
- anomaly score: -log p(machine ID) of the EMA model.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Any, ClassVar

import numpy as np
import torch
import torchaudio
from torch import nn

from noisemismatch.data.dataset import ClipSet
from noisemismatch.models.base import (
    AnomalyDetector,
    batch_indices,
    register,
    training_steps,
)

logger = logging.getLogger(__name__)

# MobileFaceNets, Table 1: (expansion t, channels c, repeats n, stride s).
MOBILEFACENET: tuple[tuple[int, int, int, int], ...] = (
    (2, 64, 5, 2),
    (4, 128, 1, 2),
    (2, 128, 6, 1),
    (4, 128, 1, 2),
    (2, 128, 2, 1),
)
# Three sequences of the huangswt/OS-SCL repository (fact recorded on 2026-09-30,
# docs/deviations.md); used only by arm B of the DCASE 2020 diagnostic.
MOBILEFACENET_REPOSITORY: tuple[tuple[int, int, int, int], ...] = (
    (2, 128, 2, 2),
    (4, 128, 2, 2),
    (4, 128, 2, 2),
)
MOBILEFACENET_SETTINGS: dict[str, tuple[tuple[int, int, int, int], ...]] = {
    "paper": MOBILEFACENET,
    "repository": MOBILEFACENET_REPOSITORY,
}


class TgramNet(nn.Module):
    """Tgram (STgram-MFN, Table 1): large-kernel 1D convolution, three CNN blocks.

    Layer normalization over the time axis and LeakyReLU slope 0.2 are facts of
    the OS-SCL repository (docs/deviations.md).
    """

    def __init__(self, mels: int, window: int, hop: int, frames: int) -> None:
        super().__init__()
        self.extractor = nn.Conv1d(1, mels, window, hop, window // 2, bias=False)
        self.blocks = nn.Sequential(
            *[
                nn.Sequential(
                    nn.LayerNorm(frames),
                    nn.LeakyReLU(0.2),
                    nn.Conv1d(mels, mels, 3, 1, 1, bias=False),
                )
                for _ in range(3)
            ]
        )

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """``(B, L)`` -> ``(B, mels, frames)``."""
        return self.blocks(self.extractor(wave.unsqueeze(1)))


class ConvBlock1d(nn.Module):
    """Two 1D convolutions with batch normalization and ReLU (OS-SCL, Table I).

    Kernel 3; padding 1, then padding 2 with dilation 2.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, 1, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, 3, 1, padding=2, dilation=2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies both convolutions."""
        return self.body(x)


class AdaptiveMaxPool1d(nn.Module):
    """Adaptive max pooling with a deterministic backward pass.

    Same windows and values as ``nn.AdaptiveMaxPool1d`` (window i covers
    ``[floor(i L / n), ceil((i + 1) L / n))``), computed by indexing, whose
    gradient accumulation has a deterministic CUDA implementation.
    """

    def __init__(self, output_size: int) -> None:
        super().__init__()
        self.output_size = output_size
        self._cache: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}

    def _windows(
        self, length: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (length, str(device))
        if key not in self._cache:
            n = self.output_size
            starts = [(i * length) // n for i in range(n)]
            ends = [-((-(i + 1) * length) // n) for i in range(n)]
            width = max(e - s for s, e in zip(starts, ends, strict=True))
            offsets = torch.arange(width)
            index = torch.tensor(starts)[:, None] + offsets[None, :]
            valid = index < torch.tensor(ends)[:, None]
            self._cache[key] = (
                index.clamp(max=length - 1).to(device),
                valid.to(device),
            )
        return self._cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, L)`` -> ``(B, C, output_size)``."""
        index, valid = self._windows(x.shape[-1], x.device)
        windows = x[..., index]  # (B, C, n, width)
        return windows.masked_fill(~valid, float("-inf")).amax(dim=-1)


class TFgramNet(nn.Module):
    """TFgram (OS-SCL, Table I): time-frequency features learned from the waveform."""

    def __init__(self, frames: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, 11, 5, 5, bias=False), nn.BatchNorm1d(64), nn.ReLU()
        )
        self.block1 = ConvBlock1d(64, 64)
        self.pool1 = nn.MaxPool1d(4)
        self.block2 = ConvBlock1d(64, 64)
        self.pool2 = AdaptiveMaxPool1d(2 * frames)
        self.block3 = ConvBlock1d(64, 128)
        self.pool3 = AdaptiveMaxPool1d(frames)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """``(B, L)`` -> ``(B, 128, frames)``."""
        x = self.pool1(self.block1(self.stem(wave.unsqueeze(1))))
        x = self.pool2(self.block2(x))
        return self.pool3(self.block3(x))


class ConvBNPReLU(nn.Module):
    """Convolution (no bias), batch norm, optional PReLU (MobileFaceNets)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int | tuple[int, int],
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel, stride, padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU(out_ch) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolution, normalization, activation."""
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """MobileNetV2 inverted residual as used by MobileFaceNets (PReLU)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int, expansion: int) -> None:
        super().__init__()
        hidden = in_ch * expansion
        self.residual = stride == 1 and in_ch == out_ch
        self.body = nn.Sequential(
            ConvBNPReLU(in_ch, hidden, 1),
            ConvBNPReLU(hidden, hidden, 3, stride, 1, groups=hidden),
            ConvBNPReLU(hidden, out_ch, 1, act=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inverted residual block."""
        y = self.body(x)
        return x + y if self.residual else y


class MobileFaceNet(nn.Module):
    """MobileFaceNet (Chen et al. 2018, Table 1).

    The global depthwise convolution is sized to the last feature map (8 x 20 for
    128 x 313 inputs).
    """

    def __init__(
        self,
        in_ch: int,
        height: int,
        width: int,
        dim: int = 128,
        setting: tuple[tuple[int, int, int, int], ...] = MOBILEFACENET,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            ConvBNPReLU(in_ch, 64, 3, 2, 1),
            ConvBNPReLU(64, 64, 3, 1, 1, groups=64),
        ]
        channels, h, w = 64, math.ceil(height / 2), math.ceil(width / 2)
        for t, c, n, s in setting:
            for k in range(n):
                stride = s if k == 0 else 1
                layers.append(Bottleneck(channels, c, stride, t))
                channels = c
                h, w = math.ceil(h / stride), math.ceil(w / stride)
        layers += [
            ConvBNPReLU(channels, 512, 1),
            ConvBNPReLU(512, 512, (h, w), groups=512, act=False),  # linear GDConv
            ConvBNPReLU(512, dim, 1, act=False),  # linear 1x1 convolution
        ]
        self.body = nn.Sequential(*layers)
        self.map_size = (h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, H, W)`` -> ``(B, dim)``."""
        return self.body(x).flatten(1)


class PerturbationHead(nn.Module):
    """Feature perturbation head: 128 -> R -> 128, then L2 normalization.

    Paper: first encoder layer to R, last decoder layer back, ReLU (TFSTgram).
    Batch normalization after each activation: fact of the OS-SCL repository.
    """

    def __init__(self, dim: int = 128, reduced: int = 64) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(dim, reduced),
            nn.ReLU(),
            nn.BatchNorm1d(reduced),
            nn.Linear(reduced, dim),
            nn.ReLU(),
            nn.BatchNorm1d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perturbed, L2-normalized features."""
        return nn.functional.normalize(self.body(x), dim=1)


class ArcFace(nn.Module):
    """ArcFace logits s * cos(theta + m * y) (margin only on the given class)."""

    def __init__(self, dim: int, n_classes: int, scale: float, margin: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_classes, dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale, self.margin = scale, margin

    def cosines(self, z: torch.Tensor) -> torch.Tensor:
        """Cosine similarities to the class centers."""
        return (
            nn.functional.normalize(z, dim=1)
            @ nn.functional.normalize(self.weight, dim=1).T
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor | None) -> torch.Tensor:
        """Scaled logits; no margin when ``labels`` is None (inference)."""
        cos = self.cosines(z)
        if labels is None:
            return self.scale * cos
        theta = torch.arccos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
        margin = self.margin * nn.functional.one_hot(labels, cos.shape[1])
        return self.scale * torch.cos(theta + margin)


def ema_decay(window_fraction: float, steps: int) -> float:
    """EMA decay whose window (1 / (1 - decay)) is a fraction of the run's steps.

    The authors' decay, 0.9999, is a window of 10,000 steps, i.e. 6.2 % of their
    161,700 training steps; the same fraction is kept here.

    Args:
        window_fraction: Window as a fraction of the number of steps.
        steps: Number of optimization steps of the run.

    Returns:
        The decay (0 if the window is shorter than one step).
    """
    return 1.0 - 1.0 / max(window_fraction * steps, 1.0)


def noisy_arcmix(
    logits: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float
) -> torch.Tensor:
    """Noisy-ArcMix loss (OS-SCL Eq. (2); Choi et al. Eq. (4)).

    The logits carry the margin of ``y_a``; target = lam * y_a + (1 - lam) * y_b.
    """
    log_p = torch.log_softmax(logits, dim=1)
    idx = torch.arange(len(logits), device=logits.device)
    return -(lam * log_p[idx, y_a] + (1 - lam) * log_p[idx, y_b]).mean()


def noise_supervised_contrastive(
    z: torch.Tensor, labels: torch.Tensor, tau: float
) -> torch.Tensor:
    """OS-SCL Eq. (1), averaged over the anchors that have a positive.

    Args:
        z: L2-normalized features ``(B, D)`` (of mixed-up inputs).
        labels: Original labels ``y_a``: positives share them with the anchor.
        tau: Temperature.
    """
    sim = z @ z.T / tau
    b = len(z)
    eye = torch.eye(b, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, float("-inf"))  # A(i) excludes the anchor
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    positives = (labels[:, None] == labels[None, :]) & ~eye
    count = positives.sum(dim=1)
    keep = count > 0
    total = torch.where(positives, log_prob, torch.zeros_like(log_prob)).sum(dim=1)
    return -(total[keep] / count[keep]).mean()


class OSSCLNet(nn.Module):
    """Features (log-Mel, Tgram, TFgram), MobileFaceNet, head, ArcFace."""

    def __init__(self, c: dict[str, Any], length: int, n_classes: int) -> None:
        super().__init__()
        mels, window, hop = int(c["n_mels"]), int(c["n_fft"]), int(c["hop"])
        frames = length // hop + 1
        # Optional keys, absent from the study configuration (arm B of the DCASE
        # 2020 diagnostic, docs/validation.md): log-Mel window, log scale and
        # MobileFaceNet sequences of the repository. The Tgram keeps ``n_fft``.
        self.logmel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=int(c.get("logmel_n_fft") or window),
            hop_length=hop,
            n_mels=mels,
            power=2.0,
        )
        self.log_eps = float(c["log_eps"])
        self.log_scale = str(c.get("log_scale", "ln"))
        if self.log_scale not in ("ln", "db"):
            raise ValueError(f"unknown log_scale {self.log_scale!r}")
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
        self.tgram = TgramNet(mels, window, hop, frames)
        self.tfgram = TFgramNet(frames)
        self.mfn = MobileFaceNet(
            3,
            mels,
            frames,
            int(c["embedding_dim"]),
            MOBILEFACENET_SETTINGS[str(c.get("mobilefacenet", "paper"))],
        )
        self.head = PerturbationHead(int(c["embedding_dim"]), int(c["fph_dim"]))
        self.arcface = ArcFace(
            int(c["embedding_dim"]),
            n_classes,
            float(c["arcface_scale"]),
            float(c["arcface_margin"]),
        )

    def log_mel(self, wave: torch.Tensor) -> torch.Tensor:
        """Log-Mel spectrogram ``(B, mels, frames)`` (natural log, STgram-MFN Eq. (1)).

        With ``log_scale: db``: 10 log10 of the power, torchaudio defaults.
        """
        if self.log_scale == "db":
            return self.to_db(self.logmel(wave))
        return torch.log(self.logmel(wave) + self.log_eps)

    def features(self, wave: torch.Tensor, mel: torch.Tensor) -> torch.Tensor:
        """Perturbed normalized features of waveforms and their log-Mel spectrograms."""
        x = torch.stack([mel, self.tgram(wave), self.tfgram(wave)], dim=1)
        return self.head(self.mfn(x))


@register
class OSSCLDetector(AnomalyDetector):
    """OS-SCL, TFSTgram variant (see module docstring).

    One model for all machine types; classes = machine type x machine ID.
    """

    name: ClassVar[str] = "osscl"
    training_scope: ClassVar[str] = "all"

    def __init__(self, config: dict[str, Any], seed: int, device: torch.device) -> None:
        super().__init__(config, seed, device)
        self.net: OSSCLNet | None = None
        self.ema: OSSCLNet | None = None
        self.classes: list[str] = []
        self.history: list[dict[str, float]] = []
        self.steps = 0
        # Diagnostics (not used in the study): clips on which the per-class
        # accuracy of the trained and EMA weights is recorded every
        # ``diagnostic_every`` epochs. Evaluation draws no random number, so the
        # training trajectory is unchanged.
        self.diagnostic_clips: ClipSet | None = None
        self.diagnostic_every = 0
        self.diagnostic_dir: Any = None  # weights saved there at each record
        self.ema_decay = 0.0
        self.diagnostics: dict[str, Any] = {}

    @staticmethod
    def class_labels(table: Any) -> np.ndarray:
        """Class of each clip: machine type / machine ID."""
        return (table["machine_type"] + "/" + table["machine_id"]).to_numpy()

    def fit(self, train_set: ClipSet) -> None:
        """Trains the network (and its EMA copy) on normal clips of all types.

        Args:
            train_set: Normal training clips of every machine type.
        """
        c = self.config
        rng = np.random.default_rng(self.seed)
        labels = self.class_labels(train_set.table)
        self.classes = sorted(set(labels))
        index = {name: k for k, name in enumerate(self.classes)}
        y_all = torch.tensor([index[v] for v in labels])
        raw = train_set.raw_waveforms()
        self.net = OSSCLNet(c, raw.shape[1], len(self.classes)).to(self.device)
        self.ema = copy.deepcopy(self.net).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=float(c["learning_rate"]),
            weight_decay=float(c["weight_decay"]),
        )
        epochs, batch = int(c["epochs"]), int(c["batch_size"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=float(c["eta_min_ratio"]) * float(c["learning_rate"]),
        )
        n_snrs = int(train_set.table["snr_db"].nunique())
        self.steps = training_steps(len(raw) // n_snrs, batch, epochs, drop_last=False)
        if c.get("ema_decay") is not None:  # arm B of the DCASE 2020 diagnostic
            self.ema_decay = float(c["ema_decay"])
        else:
            self.ema_decay = ema_decay(float(c["ema_window_fraction"]), self.steps)
        steps_per_epoch = max(self.steps // epochs, 1)
        self.net.train()
        running = 0.0
        for step, idx in enumerate(batch_indices(len(raw), batch, self.steps, rng), 1):
            loss = self._step(raw[idx], y_all[idx], rng)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self._update_ema()
            running += float(loss.detach())
            if step % steps_per_epoch == 0:
                scheduler.step()
                record = {
                    "epoch": step // steps_per_epoch,
                    "loss": running / steps_per_epoch,
                }
                epoch = step // steps_per_epoch
                if self.diagnostic_clips is not None and (
                    epoch % self.diagnostic_every == 0 or epoch == epochs
                ):
                    for name, net in (("raw", self.net), ("ema", self.ema)):
                        record[f"class_accuracy_{name}"] = self.class_accuracy(
                            net, self.diagnostic_clips
                        )
                    if self.diagnostic_dir is not None:
                        torch.save(
                            {
                                "raw": self.net.state_dict(),
                                "ema": self.ema.state_dict(),
                                "classes": self.classes,
                                "epoch": epoch,
                            },
                            self.diagnostic_dir / f"weights_epoch{epoch:03d}.pt",
                        )
                    self.net.train()
                self.history.append(record)
                logger.info("OS-SCL %s", record)
                running = 0.0
        # Safeguard recorded with every run: the EMA model must have converged to
        # the trained one on the training clips (docs/deviations.md, O-EMA).
        self.diagnostics = {
            "steps": self.steps,
            "ema_decay": self.ema_decay,
            "train_id_accuracy_raw": self.class_accuracy(self.net, train_set),
            "train_id_accuracy_ema": self.class_accuracy(self.ema, train_set),
        }
        logger.info("OS-SCL safeguard %s", self.diagnostics)

    def save_weights(self, path: Any) -> None:
        """Saves the trained and EMA weights."""
        torch.save(
            {
                "raw": self.net.state_dict(),
                "ema": self.ema.state_dict(),
                "classes": self.classes,
            },
            path,
        )

    def _step(
        self, raw: np.ndarray, y_a: torch.Tensor, rng: np.random.Generator
    ) -> torch.Tensor:
        c = self.config
        dev = self.device
        wave = torch.from_numpy(raw).to(dev, torch.float32) / 32768.0
        y_a = y_a.to(dev)
        lam = float(rng.beta(float(c["mixup_alpha"]), float(c["mixup_alpha"])))
        perm = torch.from_numpy(rng.permutation(len(raw))).to(dev)
        mel = self.net.log_mel(wave)
        wave_mix = lam * wave + (1 - lam) * wave[perm]
        mel_mix = lam * mel + (1 - lam) * mel[perm]
        z = self.net.features(wave_mix, mel_mix)
        logits = self.net.arcface(z, y_a)
        y_b = y_a[perm]
        return noise_supervised_contrastive(
            z, y_a, float(c["temperature"])
        ) + noisy_arcmix(logits, y_a, y_b, lam)

    @torch.no_grad()
    def _update_ema(self) -> None:
        """EMA of the weights and running statistics.

        Also applies the weight decay step of the repository's EMA to the trained
        weights (docs/deviations.md).
        """
        decay, extra = self.ema_decay, float(self.config["ema_weight_decay"])
        for p_ema, p in zip(
            self.ema.state_dict().values(), self.net.state_dict().values(), strict=True
        ):
            if p_ema.dtype.is_floating_point:
                p_ema.mul_(decay).add_(p, alpha=1 - decay)
                p.mul_(1 - extra)
            else:
                p_ema.copy_(p)

    def embeddings(self, net: OSSCLNet, clips: ClipSet) -> np.ndarray:
        """Normalized features of clips with the given network, in evaluation mode."""
        raw = clips.raw_waveforms()
        net.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(raw), 64):
                wave = (
                    torch.from_numpy(raw[start : start + 64]).to(
                        self.device, torch.float32
                    )
                    / 32768.0
                )
                out.append(net.features(wave, net.log_mel(wave)).cpu().numpy())
        return np.concatenate(out)

    def probabilities(self, net: OSSCLNet, clips: ClipSet) -> np.ndarray:
        """Class probabilities (no margin) of the given network, ``(n, n_classes)``."""
        z = torch.from_numpy(self.embeddings(net, clips)).to(self.device)
        with torch.no_grad():
            return torch.softmax(net.arcface(z, None), dim=1).cpu().numpy()

    def class_accuracy(self, net: OSSCLNet, clips: ClipSet) -> dict[str, float]:
        """Accuracy of the machine-ID classification per class."""
        predicted = self.probabilities(net, clips).argmax(axis=1)
        index = {name: k for k, name in enumerate(self.classes)}
        true = np.array([index[v] for v in self.class_labels(clips.table)])
        return {
            name: float((predicted[true == k] == k).mean())
            for name, k in index.items()
            if (true == k).any()
        }

    def id_probabilities(self, clips: ClipSet) -> np.ndarray:
        """Class probabilities of the EMA model (no margin), ``(n, 1, n_classes)``."""
        if self.ema is None:
            raise RuntimeError("fit must be called first")
        return self.probabilities(self.ema, clips)[:, None, :]

    def score(self, test_set: ClipSet) -> np.ndarray:
        """-log p(machine ID of the clip) of the EMA model (no margin).

        With ``score_margin: true`` (arm B of the DCASE 2020 diagnostic), the
        logit of the clip's ID carries the ArcFace margin: cross-entropy with
        the margin.

        Args:
            test_set: Test clips.

        Returns:
            Float array shaped ``(len(test_set),)``.
        """
        if self.ema is None:
            raise RuntimeError("fit must be called first")
        index = {name: k for k, name in enumerate(self.classes)}
        labels = torch.tensor([index[v] for v in self.class_labels(test_set.table)])
        raw = test_set.raw_waveforms()
        out = []
        with torch.no_grad():
            for start in range(0, len(raw), 64):
                wave = (
                    torch.from_numpy(raw[start : start + 64]).to(
                        self.device, torch.float32
                    )
                    / 32768.0
                )
                z = self.ema.features(wave, self.ema.log_mel(wave))
                y = labels[start : start + 64].to(self.device)
                margin_labels = y if self.config.get("score_margin", False) else None
                log_p = torch.log_softmax(self.ema.arcface(z, margin_labels), dim=1)
                out.append(-log_p[torch.arange(len(y), device=self.device), y])
        return torch.cat(out).cpu().numpy().astype(float)
