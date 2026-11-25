import math

import numpy as np
import pandas as pd
import pytest
import torch

from noisemismatch.config import CONFIG_DIR, load_yaml
from noisemismatch.data.dataset import ClipSet
from noisemismatch.models.wilkinghoff import (
    SpectrogramNet,
    SpectrumNet,
    SubClusterAdaCos,
    WilkinghoffDetector,
    WilkinghoffNet,
    adjust_length,
    exchange_labels,
    same_padding,
    spectrogram,
    spectrum,
    statex,
)

BN = {"eps": 1e-3, "momentum": 1e-2}
CPU = torch.device("cpu")


def test_input_sizes_match_the_papers():
    x = torch.randn(2, 160000)
    assert spectrogram(x, 1024, 512).shape == (2, 1, 311, 513)  # D23 Fig. 1
    assert spectrum(x).shape == (2, 80000)
    assert spectrogram(torch.randn(1, 288000), 1024, 512).shape == (1, 1, 561, 513)


def test_temporal_mean_normalization():
    s = spectrogram(torch.randn(1, 16000), 1024, 512)
    assert torch.allclose(s.mean(dim=2), torch.zeros(1, 1, 513), atol=1e-5)


def test_spectrogram_branch_sizes_of_thesis_table_15():
    net = SpectrogramNet(513, 128, BN).eval()
    x = torch.randn(2, 1, 311, 513)
    stem = net.pool(net.stem(x))
    assert net.stem(x).shape[2:] == (156, 257)
    assert stem.shape[2:] == (77, 128)
    sizes = []
    for block in net.blocks:
        stem = block(stem)
        sizes.append(tuple(stem.shape[1:]))
    assert sizes[1::2] == [(16, 77, 128), (32, 39, 64), (64, 20, 32), (128, 10, 16)]
    assert net.flat_dim == 16 * 128
    assert net(x).shape == (2, 128)


def test_spectrum_branch_sizes_of_thesis_table_14():
    net = SpectrumNet(80000, [256, 64, 16], [64, 32, 4], 128, 128, BN).eval()
    h = torch.randn(2, 80000).unsqueeze(1)
    sizes = []
    for conv in net.convs:
        h = conv(h)
        sizes.append(h.shape[2])
    assert sizes == [1250, 40, 10]
    assert net.flat_dim == 1280
    assert net(torch.randn(2, 80000)).shape == (2, 128)


def test_no_bias_anywhere():
    net = WilkinghoffNet(160000, load_yaml(CONFIG_DIR / "models" / "wilkinghoff.yaml"))
    for m in net.modules():
        if isinstance(m, torch.nn.Conv1d | torch.nn.Conv2d | torch.nn.Linear):
            assert m.bias is None


def test_same_padding_like_tensorflow():
    assert same_padding(128, 3, 2) == (0, 1)  # extra padding at the end
    assert same_padding(77, 3, 2) == (1, 1)
    assert same_padding(80000, 256, 64) == (96, 96)


def test_subcluster_adacos_initial_scale_and_loss_by_hand():
    loss = SubClusterAdaCos(n_classes=2, n_sub=2, dim=2, trainable=False)
    assert float(loss.scale) == pytest.approx(math.sqrt(2) * math.log(3))
    with torch.no_grad():  # centers: class 0 -> (1,0),(0,1); class 1 -> (-1,0),(0,-1)
        loss.centers.copy_(torch.tensor([[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, -1.0]]))
    e = torch.tensor([[1.0, 0.0]])
    y = torch.tensor([[1.0, 0.0]])
    s = float(loss.scale)
    value = loss(e, y, update=False)
    # logits s*(1, 0, -1, 0); p(class 0) = (e^s + 1) / (e^s + 1 + e^-s + 1)
    expected = -math.log((math.exp(s) + 1) / (math.exp(s) + 2 + math.exp(-s)))
    assert float(value) == pytest.approx(expected, rel=1e-6)


def test_subcluster_adacos_scale_update_follows_definition_4():
    loss = SubClusterAdaCos(n_classes=2, n_sub=2, dim=2, trainable=False)
    with torch.no_grad():
        loss.centers.copy_(torch.tensor([[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, -1.0]]))
    e = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    s0 = float(loss.scale)
    cos = loss.cosines(e)
    loss.update_scale(cos, y)
    logits = s0 * cos
    f_max = float(logits.max())
    b_avg = float(torch.exp(logits - f_max).sum(dim=1).mean())
    # angles to the closest sub-cluster: sample 0: (0, pi); sample 1: (pi/4, 3pi/4)
    mixed = [0.0, 0.5 * math.pi / 4 + 0.5 * 3 * math.pi / 4]
    theta_med = float(torch.median(torch.tensor(mixed)))
    expected = (f_max + math.log(b_avg)) / math.cos(min(math.pi / 4, theta_med))
    assert float(loss.scale) == pytest.approx(expected, rel=1e-5)


def test_statex_exchanges_frame_statistics():
    rng = torch.Generator().manual_seed(0)
    spec = torch.randn(2, 1, 5, 7, generator=rng) * torch.tensor([1.0, 3.0]).view(
        2, 1, 1, 1
    )
    out = statex(spec, torch.tensor([1, 0]), eps=0.0)
    assert torch.allclose(out[0].mean(dim=2), spec[1].mean(dim=2), atol=1e-5)
    assert torch.allclose(out[0].std(dim=2), spec[1].std(dim=2), atol=1e-5)


def test_exchange_labels_triple_the_classes():
    y = torch.eye(2)
    out = exchange_labels(y, torch.tensor([1, 0]), torch.tensor([True, False]))
    assert out.tolist() == [[0, 0, 0.5, 0, 0, 0.5], [0, 1, 0, 0, 0, 0]]


def test_adjust_length_repeats_and_crops():
    rng = np.random.default_rng(0)
    wave = np.arange(5)
    out = adjust_length(wave, 12, rng)
    assert len(out) == 12
    assert set(np.diff(out)) <= {1, -4}  # a crop of the repeated signal
    assert adjust_length(wave, 5, rng) is wave


class _Audio:
    fingerprint = "test-wilkinghoff"

    def __init__(self, n=24, length=16000):
        rng = np.random.default_rng(0)
        self.table = pd.DataFrame(
            {
                "file_machine_id": [f"id_0{k % 2}" for k in range(n)],
                "file_label": "normal",
                "file_clip": [f"{k:08d}" for k in range(n)],
            }
        )
        t = np.arange(length) / 16000
        self.waves = np.stack(
            [
                (
                    3000 * np.sin(2 * np.pi * (500 + 1500 * (k % 2)) * t)
                    + rng.normal(0, 300, length)
                ).astype(np.int16)
                for k in range(n)
            ]
        )

    def load(self, machine_type, snr, files):
        rows = {r: k for k, r in enumerate(self.table["file_clip"])}
        return self.waves[[rows[c] for c in files["file_clip"]]]


def test_detector_trains_and_scores_on_a_tiny_set():
    audio = _Audio()
    table = pd.DataFrame(
        {
            "machine_type": "fan",
            "machine_id": audio.table["file_machine_id"],
            "snr_db": 0,
            **{c: audio.table[c] for c in audio.table.columns},
        }
    )
    config = {
        **load_yaml(CONFIG_DIR / "models" / "wilkinghoff.yaml"),
        "input_length": 16000,
        "epochs": 2,
        "batch_size": 8,
        "kmeans_clusters": 2,
    }
    torch.manual_seed(0)
    model = WilkinghoffDetector(config, 0, CPU)
    model.fit(ClipSet(table, audio))
    assert len(model.history) == 2
    scores = model.score(ClipSet(table, audio))
    assert scores.shape == (len(table),)
    assert np.all(scores >= -1e-6)
    assert np.all(scores <= 2.0)


def test_crop_range_of_repeated_clips():
    class Audio:
        def native_lengths(self, machine_type, snr, files):
            return np.array([6, 16, 10])

    table = pd.DataFrame(
        {
            "machine_type": "fan",
            "snr_db": 0,
            "file_machine_id": "id_00",
            "file_label": "normal",
            "file_clip": ["a", "b", "c"],
        }
    )
    config = {
        **load_yaml(CONFIG_DIR / "models" / "wilkinghoff.yaml"),
        "input_length": 16,
    }
    model = WilkinghoffDetector(config, 0, CPU)
    # 6 samples -> repeated 3 times (18): starts 0..2; 16 -> 0; 10 -> 20: 0..4
    assert model._crop_max(ClipSet(table, Audio()), stored=32).tolist() == [2, 0, 4]
    assert model._crop_max(ClipSet(table, Audio()), stored=16).tolist() == [0, 0, 0]
