import numpy as np
import pandas as pd
import torch

from noisemismatch.data.dataset import ClipSet
from noisemismatch.models.mobilenetv2 import (
    KerasMobileNetV2,
    MobileNetV2Detector,
    SamePad,
    make_divisible,
    window_images,
)
from noisemismatch.utils.repro import seed_everything

SAMPLES = 8000
CPU = torch.device("cpu")
CONFIG = {
    "features": {"n_mels": 32, "n_fft": 512, "hop_length": 256, "log_floor": "max"},
    "n_frames": 16,
    "n_hop_frames": 4,
    "alpha": 0.5,
    "bn_eps": 1e-3,
    # Keras' momentum (PyTorch 1e-3) needs thousands of steps for the running
    # statistics to settle; this tiny test trains for a few dozen.
    "bn_momentum": 0.1,
    "epochs": 8,
    "batch_size": 16,
    "learning_rate": 1e-3,
    "adam_betas": [0.9, 0.999],
    "adam_eps": 1e-7,
    "validation_split": 0.0,
}
TONES = {"id_00": 800, "id_02": 2400}


def test_make_divisible_matches_keras():
    assert make_divisible(32 * 0.5) == 16
    assert make_divisible(int(24 * 0.5)) == 16  # 12 -> 16 (not below 0.9 x 12)
    assert make_divisible(int(320 * 0.5)) == 160


def test_same_padding_is_asymmetric_for_even_sizes():
    padded = SamePad(3, 2)(torch.ones(1, 1, 4, 6))
    assert padded.shape == (1, 1, 5, 7)  # one row/column added, at the end
    assert padded[0, 0, -1].sum() == 0
    assert padded[0, 0, 0].sum() == 6


def test_window_images_layout():
    # features[c, m, t] = 100 * m + t, 3 bands, 6 frames
    features = (100 * np.arange(3)[:, None] + np.arange(6)[None, :])[None].astype(float)
    images = window_images(features, n_frames=4, hop=2)
    assert images.shape == (1, 2, 4, 3)  # windows starting at frames 0 and 2
    assert images[0, 1, 0].tolist() == [2, 102, 202]  # frame 2, all bands
    assert images[0, 1, 3].tolist() == [5, 105, 205]


def test_network_output_shape():
    assert KerasMobileNetV2(3)(torch.zeros(2, 1, 64, 128)).shape == (2, 3)


class _Audio:
    """Each machine ID hums at its own frequency; anomalies hum at 5000 Hz."""

    fingerprint = "test-mobilenetv2-audio"

    def __init__(self):
        rows = []
        for machine_id in TONES:
            rows += [(machine_id, "normal", f"{k:08d}") for k in range(12)]
            rows += [(machine_id, "abnormal", f"{k:08d}") for k in range(4)]
        self.files = pd.DataFrame(
            rows, columns=["file_machine_id", "file_label", "file_clip"]
        )

    @staticmethod
    def _wave(snr, machine_id, label, clip):
        rng = np.random.default_rng(
            int(clip) + 97 * (label == "abnormal") + 13 * snr + len(machine_id)
        )
        freq = TONES[machine_id] if label == "normal" else 5000
        t = np.arange(SAMPLES) / 16000
        x = 0.3 * np.sin(2 * np.pi * freq * t) + 0.05 * rng.normal(size=SAMPLES)
        return (x * 32767).astype(np.int16)

    def index(self, machine_type, snr):
        return self.files

    def array(self, machine_type, snr):
        return np.stack([self._wave(snr, *r) for r in self.files.to_numpy()])

    def load(self, machine_type, snr, files):
        return np.stack([self._wave(snr, *r) for r in files.to_numpy()])


def _clips(tmp_path, audio, snrs, label=None):
    rows = []
    for snr in snrs:
        for machine_id, lab, clip in audio.files.to_numpy():
            rows.append(
                {
                    "machine_type": "fan",
                    "machine_id": machine_id,
                    "label": lab,
                    "clip": clip,
                    "snr_db": snr,
                    "file_machine_id": machine_id,
                    "file_label": lab,
                    "file_clip": clip,
                    "y": int(lab == "abnormal"),
                    "split": "train" if lab == "normal" else "test",
                    "strict_exclude": False,
                }
            )
    table = pd.DataFrame(rows)
    if label is not None:
        table = table[table["label"] == label].reset_index(drop=True)
    return ClipSet(table, audio, tmp_path)


def _fit(tmp_path, snrs=(0,), seed=0):
    seed_everything(seed, warn_only=True)
    model = MobileNetV2Detector(CONFIG, seed, CPU)
    model.fit(_clips(tmp_path, _Audio(), snrs, label="normal"))
    return model


def test_fit_classifies_ids_and_scores_foreign_sounds_higher(tmp_path):
    model = _fit(tmp_path)
    assert model.classes == ["id_00", "id_02"]
    assert len(model.history) == CONFIG["epochs"]
    clips = _clips(tmp_path, _Audio(), (0,))
    probabilities = model.id_probabilities(clips)
    n_frames_total = 1 + SAMPLES // 256
    assert probabilities.shape == (len(clips), n_frames_total - 16 + 1, 2)
    normal = clips.table["label"] == "normal"
    predicted = probabilities.mean(axis=1).argmax(axis=1)
    true = (clips.table["machine_id"] == "id_02").astype(int).to_numpy()
    assert (predicted[normal] == true[normal]).mean() == 1.0
    scores = model.score(clips)
    assert scores.shape == (len(clips),)
    assert np.median(scores[~normal]) > np.median(scores[normal])


def test_training_is_deterministic(tmp_path):
    clips = _clips(tmp_path, _Audio(), (0,))
    a = _fit(tmp_path, seed=1).score(clips)
    b = _fit(tmp_path, seed=1).score(clips)
    assert np.array_equal(a, b)


def test_multi_condition_uses_the_single_snr_step_budget(tmp_path):
    single = _fit(tmp_path, snrs=(0,))
    multi = _fit(tmp_path, snrs=(0, 6))
    assert multi.steps == single.steps
    assert multi.n_train_images == 2 * single.n_train_images


def test_committed_config_batch_norm_matches_keras():
    # keras.applications.MobileNetV2 (Keras 2.8): BatchNormalization(epsilon=1e-3,
    # momentum=0.999) in every layer; PyTorch's momentum is 1 - Keras' momentum.
    from noisemismatch.config import CONFIG_DIR, load_yaml

    c = load_yaml(CONFIG_DIR / "models" / "mobilenetv2.yaml")
    net = KerasMobileNetV2(4, c["alpha"], c["bn_eps"], c["bn_momentum"])
    layers = [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert len(layers) == 52  # as in the Keras model
    assert {(m.eps, m.momentum) for m in layers} == {(1e-3, 1e-3)}


def test_running_variance_is_unbiased_as_in_tensorflow():
    # TensorFlow's fused batch norm updates the moving variance with Bessel's
    # correction (fused_batch_norm_op.cc, v2.8.0); so does PyTorch.
    bn = torch.nn.BatchNorm2d(4, eps=1e-3, momentum=1e-3).train()
    x = torch.randn(2, 4, 2, 2, generator=torch.Generator().manual_seed(0))
    bn(x)
    unbiased = x.var((0, 2, 3), unbiased=True)
    assert torch.allclose(bn.running_var, 0.999 + 0.001 * unbiased, atol=1e-8)
