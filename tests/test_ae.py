import numpy as np
import pandas as pd
import pytest
import torch

from noisemismatch.data.dataset import ClipSet
from noisemismatch.models.ae import (
    MimiiAutoencoder,
    _Network,
    frame_vectors,
    keras_validation_split,
)
from noisemismatch.utils.repro import seed_everything

SAMPLES = 4000
CONFIG = {
    "features": {"n_mels": 16, "n_fft": 256, "hop_length": 128, "pad_mode": "reflect"},
    "frames": 5,
    "hidden_units": [16, 16, 4, 16, 16],
    "epochs": 3,
    "batch_size": 64,
    "validation_split": 0.1,
    "learning_rate": 0.01,
    "adam_betas": [0.9, 0.999],
    "adam_eps": 1e-7,
}
CPU = torch.device("cpu")


def test_frame_vectors_layout():
    # features[c, m, t] = 100 * m + t
    features = (100 * np.arange(3)[:, None] + np.arange(6)[None, :])[None].astype(float)
    vectors = frame_vectors(features, frames=2)
    assert vectors.shape == (1, 5, 6)
    # Vector s = [frame s (3 bands), frame s+1 (3 bands)].
    assert vectors[0, 0].tolist() == [0, 100, 200, 1, 101, 201]
    assert vectors[0, 4].tolist() == [4, 104, 204, 5, 105, 205]


def test_keras_validation_split():
    assert keras_validation_split(100, 0.1) == 90
    assert keras_validation_split(11, 0.1) == 9
    assert keras_validation_split(100, 0.0) == 100


def test_network_shapes_and_keras_init():
    network = _Network(320, [64, 64, 8, 64, 64])
    assert network(torch.zeros(7, 320)).shape == (7, 320)
    linears = [m for m in network.layers if isinstance(m, torch.nn.Linear)]
    assert [m.out_features for m in linears] == [64, 64, 8, 64, 64, 320]
    assert all(torch.count_nonzero(m.bias) == 0 for m in linears)


class _Audio:
    """Clips of type 'fan' = tone of a frequency per label + noise; per-clip seed."""

    fingerprint = "test-ae-audio"

    def __init__(self, n_normal=12, n_abnormal=4):
        self.files = {}
        for snr in (0, 6):
            rows = [("id_00", "normal", f"{k:08d}") for k in range(n_normal)]
            rows += [("id_00", "abnormal", f"{k:08d}") for k in range(n_abnormal)]
            self.files[snr] = pd.DataFrame(
                rows, columns=["file_machine_id", "file_label", "file_clip"]
            )

    @staticmethod
    def _wave(snr, label, clip):
        rng = np.random.default_rng(int(clip) + 1000 * (label == "abnormal") + 7 * snr)
        t = np.arange(SAMPLES) / 16000
        freq = 1000 if label == "normal" else 3000
        x = 0.3 * np.sin(2 * np.pi * freq * t) + 0.05 * rng.normal(size=SAMPLES)
        return (x * 32767).astype(np.int16)

    def index(self, machine_type, snr):
        return self.files[snr]

    def array(self, machine_type, snr):
        return np.stack(
            [self._wave(snr, r[1], r[2]) for r in self.files[snr].to_numpy()]
        )

    def load(self, machine_type, snr, files):
        return np.stack([self._wave(snr, r[1], r[2]) for r in files.to_numpy()])


def _clips(tmp_path, audio, snrs, label=None, split=None):
    rows = []
    for snr in snrs:
        for machine_id, lab, clip in audio.files[snr].to_numpy():
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


def _fit(tmp_path, audio, snrs, seed=0):
    seed_everything(seed)
    model = MimiiAutoencoder(CONFIG, seed, CPU)
    model.fit(_clips(tmp_path, audio, snrs, label="normal"))
    return model


def test_fit_score_shape_sign_and_validation_split(tmp_path):
    audio = _Audio()
    model = _fit(tmp_path, audio, (0,))
    # 12 clips x (1 + 4000 // 128 - 5 + 1) = 12 x 28 vectors, last 10 % held out.
    assert model.n_train_vectors == keras_validation_split(12 * 28, 0.1)
    assert model.n_val_vectors == 12 * 28 - model.n_train_vectors
    assert len(model.history) == CONFIG["epochs"]
    scores = model.score(_clips(tmp_path, audio, (0,)))
    assert scores.shape == (16,)
    assert np.isfinite(scores).all()
    labels = _clips(tmp_path, audio, (0,)).labels
    # Unseen tone -> larger reconstruction error.
    assert scores[labels == 1].min() > scores[labels == 0].max()


def test_training_is_deterministic(tmp_path):
    audio = _Audio()
    test = _clips(tmp_path, audio, (0,))
    a = _fit(tmp_path, audio, (0,), seed=3).score(test)
    b = _fit(tmp_path, audio, (0,), seed=3).score(test)
    c = _fit(tmp_path, audio, (0,), seed=4).score(test)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_multi_condition_uses_the_single_snr_step_budget(tmp_path):
    audio = _Audio()
    single = _fit(tmp_path, audio, (0,))
    multi = _fit(tmp_path, audio, (0, 6))
    assert multi.steps == single.steps
    assert multi.n_train_vectors == keras_validation_split(2 * 12 * 28, 0.1)


def test_score_before_fit_fails(tmp_path):
    with pytest.raises(RuntimeError, match="fit must be called"):
        MimiiAutoencoder(CONFIG, 0, CPU).score(_clips(tmp_path, _Audio(), (0,)))
