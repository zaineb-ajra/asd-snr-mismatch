import math

import numpy as np
import pandas as pd
import pytest
import torch

from noisemismatch.config import CONFIG_DIR, load_yaml
from noisemismatch.data.dataset import ClipSet
from noisemismatch.models.osscl import (
    AdaptiveMaxPool1d,
    ArcFace,
    MobileFaceNet,
    OSSCLDetector,
    OSSCLNet,
    PerturbationHead,
    TFgramNet,
    TgramNet,
    noise_supervised_contrastive,
    noisy_arcmix,
)

CPU = torch.device("cpu")
CONFIG = load_yaml(CONFIG_DIR / "models" / "osscl.yaml")


def test_tfgram_sizes_of_table_1():
    net = TFgramNet(313).eval()
    wave = torch.randn(2, 160000)
    x = net.stem(wave.unsqueeze(1))
    assert x.shape == (2, 64, 32000)  # kernel 11, stride 5, padding 5
    x = net.pool1(net.block1(x))
    assert x.shape == (2, 64, 8000)
    x = net.pool2(net.block2(x))
    assert x.shape == (2, 64, 626)
    assert net.pool3(net.block3(x)).shape == (2, 128, 313)


def test_tgram_and_logmel_sizes():
    assert TgramNet(128, 1024, 512, 313).eval()(torch.randn(2, 160000)).shape == (
        2,
        128,
        313,
    )
    net = OSSCLNet(CONFIG, 160000, 15)
    assert net.log_mel(torch.randn(2, 160000)).shape == (2, 128, 313)


def test_mobilefacenet_output_and_gdconv_size():
    net = MobileFaceNet(3, 128, 313).eval()
    assert net.map_size == (8, 20)
    assert net(torch.randn(2, 3, 128, 313)).shape == (2, 128)


def test_perturbation_head_normalizes():
    head = PerturbationHead(128, 64).eval()
    z = head(torch.randn(4, 128))
    assert torch.allclose(z.norm(dim=1), torch.ones(4), atol=1e-5)


def test_arcface_margin_only_on_the_given_class():
    arc = ArcFace(2, 2, scale=30.0, margin=0.4)
    with torch.no_grad():
        arc.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    z = torch.tensor([[1.0, 1.0]])  # 45 degrees from both centers
    logits = arc(z, torch.tensor([0]))
    assert float(logits[0, 0]) == pytest.approx(
        30 * math.cos(math.pi / 4 + 0.4), rel=1e-5
    )
    assert float(logits[0, 1]) == pytest.approx(30 * math.cos(math.pi / 4), rel=1e-5)
    assert torch.allclose(arc(z, None), 30 * arc.cosines(z))


def test_noisy_arcmix_equals_equation_2():
    logits = torch.tensor([[2.0, 0.5, -1.0]])
    lam = 0.3
    log_p = torch.log_softmax(logits, dim=1)[0]
    expected = -(lam * log_p[0] + (1 - lam) * log_p[2])
    value = noisy_arcmix(logits, torch.tensor([0]), torch.tensor([2]), lam)
    assert float(value) == pytest.approx(float(expected))


def test_contrastive_loss_by_hand():
    # three unit vectors; samples 0 and 1 share a label, sample 2 has none
    z = torch.tensor([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])
    labels = torch.tensor([0, 0, 1])
    tau = 0.5
    s = z @ z.T / tau

    def term(i, p):
        others = [a for a in range(3) if a != i]
        return -(s[i, p] - torch.logsumexp(s[i, others], dim=0))

    expected = (term(0, 1) + term(1, 0)) / 2  # anchor 2 has no positive
    value = noise_supervised_contrastive(z, labels, tau)
    assert float(value) == pytest.approx(float(expected), rel=1e-6)


def test_parameter_count_for_the_paper_comparison():
    # OS-SCL, Table V: TFSTgram (OS-SCL) has 1.38 M parameters (41 classes).
    net = OSSCLNet(CONFIG, 160000, 41)
    n = sum(p.numel() for p in net.parameters())
    assert 1.0e6 < n < 2.0e6


class _Audio:
    fingerprint = "test-osscl"

    def __init__(self, n=16, length=16000):
        rng = np.random.default_rng(0)
        self.clips = [f"{k:08d}" for k in range(n)]
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
        rows = {c: k for k, c in enumerate(self.clips)}
        return self.waves[[rows[c] for c in files["file_clip"]]]


def test_detector_trains_and_scores_on_a_tiny_set():
    audio = _Audio()
    table = pd.DataFrame(
        {
            "machine_type": "fan",
            "machine_id": [f"id_0{k % 2}" for k in range(16)],
            "snr_db": 0,
            "file_machine_id": [f"id_0{k % 2}" for k in range(16)],
            "file_label": "normal",
            "file_clip": audio.clips,
        }
    )
    config = {**CONFIG, "epochs": 2, "batch_size": 8}
    torch.manual_seed(0)
    model = OSSCLDetector(config, 0, CPU)
    model.fit(ClipSet(table, audio))
    assert len(model.history) == 2
    scores = model.score(ClipSet(table, audio))
    assert scores.shape == (16,)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0)
    assert set(model.diagnostics) >= {
        "ema_decay",
        "train_id_accuracy_raw",
        "train_id_accuracy_ema",
    }


def test_adaptive_max_pool_equals_pytorch():
    x = torch.randn(2, 3, 8000)
    for n in (626, 313, 7):
        ours = AdaptiveMaxPool1d(n)(x)
        assert torch.equal(ours, torch.nn.AdaptiveMaxPool1d(n)(x))


def test_ema_decay_is_a_fraction_of_the_number_of_steps():
    from noisemismatch.models.osscl import ema_decay

    fraction = CONFIG["ema_window_fraction"]
    # the authors' 161,700 steps give back their decay, 0.9999 (10,000 steps)
    assert ema_decay(fraction, 161700) == pytest.approx(0.9999, abs=1e-7)
    # the main split of the study: 41,400 steps -> window of 2,560 steps
    assert ema_decay(fraction, 41400) == pytest.approx(1 - 1 / 2560.3, abs=1e-7)
    assert round(ema_decay(fraction, 41400), 5) == 0.99961
    assert ema_decay(fraction, 4) == 0.0  # window shorter than one step


ARM_B = {  # arm B of the DCASE 2020 diagnostic (configs/dcase2020_check.yaml)
    "logmel_n_fft": 2048,
    "log_scale": "db",
    "mobilefacenet": "repository",
    "score_margin": True,
    "ema_decay": 0.9999,
}


def test_arm_b_network_uses_the_repository_values():
    net = OSSCLNet({**CONFIG, **ARM_B}, 160000, 41).eval()
    assert net.logmel.n_fft == 2048
    assert net.tgram.extractor.kernel_size == (1024,)  # the Tgram keeps n_fft
    assert net.mfn.map_size == (8, 20)
    # docs/deviations.md: 1.30 M parameters with the repository's configuration
    assert round(sum(p.numel() for p in net.parameters()) / 1e6, 2) == 1.30
    wave = torch.randn(2, 160000) * 0.1
    power = net.logmel(wave)
    expected = 10 * torch.log10(power.clamp(min=1e-10))
    assert torch.allclose(net.log_mel(wave), expected, atol=1e-4)


def test_arm_b_detector_fixed_decay_and_margin_score():
    audio = _Audio()
    table = pd.DataFrame(
        {
            "machine_type": "fan",
            "machine_id": [f"id_0{k % 2}" for k in range(16)],
            "snr_db": 0,
            "file_machine_id": [f"id_0{k % 2}" for k in range(16)],
            "file_label": "normal",
            "file_clip": audio.clips,
        }
    )
    config = {**CONFIG, **ARM_B, "epochs": 2, "batch_size": 8}
    torch.manual_seed(0)
    model = OSSCLDetector(config, 0, CPU)
    model.fit(ClipSet(table, audio))
    assert model.diagnostics["ema_decay"] == 0.9999
    with_margin = model.score(ClipSet(table, audio))
    model.config = {**config, "score_margin": False}
    without = model.score(ClipSet(table, audio))
    # the margin lowers the logit of the clip's own ID: the cross-entropy grows
    assert np.all(with_margin > without)
