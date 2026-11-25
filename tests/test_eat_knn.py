import numpy as np
import pandas as pd
import pytest
import torch

from noisemismatch.data.dataset import ClipSet
from noisemismatch.features.eat import POOLINGS, filterbank, pool
from noisemismatch.features.eat_encoder import EATEncoder
from noisemismatch.models.eat_knn import EatKnnDetector, knn_scores
from noisemismatch.verification_eat import compare, fidelity_clips

CPU = torch.device("cpu")


def test_pool_keeps_frequency_positions_and_averages_time():
    # tokens[b, 1 + t * 8 + f, d] = 100 * f + t (time-major patch order)
    n_time, dim = 4, 768
    t = np.arange(n_time)[:, None, None]
    f = np.arange(8)[None, :, None]
    patches = np.broadcast_to(100.0 * f + t, (n_time, 8, dim)).reshape(1, -1, dim)
    cls = np.full((1, 1, dim), -7.0)
    tokens = torch.from_numpy(np.concatenate([cls, patches], axis=1))
    out = pool(tokens).numpy()[0]
    assert out.shape == (8 * dim + dim,)
    time_mean = out[POOLINGS["time_mean"]].reshape(8, dim)
    assert np.allclose(time_mean[:, 0], 100.0 * np.arange(8) + 1.5)
    assert np.all(out[POOLINGS["cls"]] == -7.0)


def test_knn_scores_by_hand():
    reference = np.array([[1.0, 0.0], [0.0, 1.0]])
    queries = np.array([[1.0, 1.0], [2.0, 0.0]])
    k1 = knn_scores(reference, queries, 1, CPU)
    assert k1 == pytest.approx([1 - np.sqrt(0.5), 0.0])
    k2 = knn_scores(reference, queries, 2, CPU)
    assert k2 == pytest.approx([1 - np.sqrt(0.5), 0.5])
    with pytest.raises(ValueError, match="exceeds"):
        knn_scores(reference, queries, 3, CPU)


def test_filterbank_pads_to_target_length():
    wave = torch.from_numpy(
        np.random.default_rng(0).normal(0, 0.1, (1, 160000)).astype(np.float32)
    )
    x = filterbank(wave, 128, 10, 1024, -4.268, 4.569)
    assert x.shape == (1, 1, 1024, 128)
    # 10 s at 16 kHz: 998 frames; the 26 padded frames hold a normalized zero
    assert torch.allclose(x[0, 0, 998:], torch.tensor(4.268 / (2 * 4.569)))
    assert not torch.allclose(x[0, 0, 997], torch.tensor(4.268 / (2 * 4.569)))


def test_encoder_tensor_names_match_the_checkpoint_layout():
    names = dict(EATEncoder().state_dict())
    assert len(names) == 150
    assert names["fixed_positional_encoder.positions"].shape == (1, 6144, 768)
    assert names["local_encoder.proj.weight"].shape == (768, 1, 16, 16)
    assert names["blocks.11.mlp.fc2.weight"].shape == (768, 3072)
    out = EATEncoder().eval()(torch.zeros(2, 1, 32, 128))
    assert out.shape == (2, 1 + 2 * 8, 768)


def test_detector_scores_with_cached_embeddings(monkeypatch):
    config = {
        "checkpoint": {"repo": "r/m", "revision": "x", "file": "f", "sha256": "s"},
        "features": {"batch_size": 2},
        "pooling": "time_mean",
        "k": 1,
        "metric": "cosine",
    }
    rng = np.random.default_rng(1)
    train = rng.normal(size=(5, 6912)).astype(np.float32)
    test = np.concatenate([train[:1], rng.normal(size=(2, 6912))]).astype(np.float32)
    arrays = {5: train, 3: test}
    monkeypatch.setattr(ClipSet, "features", lambda self, extractor: arrays[len(self)])
    clips = lambda n: ClipSet(pd.DataFrame({"i": range(n)}), audio=None)  # noqa: E731
    model = EatKnnDetector(config, 0, CPU)
    model.fit(clips(5))
    scores = model.score(clips(3))
    assert scores[0] == pytest.approx(0.0, abs=1e-12)  # identical to a reference clip
    assert np.all(scores[1:] > 0.5)
    variants = model.score_variants(clips(3))
    assert set(variants) == {("time_mean", 1), ("time_mean", 5), ("cls", 1), ("cls", 5)}


def test_compare_applies_the_fixed_criterion():
    rng = np.random.default_rng(2)
    ref = {
        "fbank": rng.normal(size=(2, 4, 3)),
        "time_mean": rng.normal(size=(2, 10)),
        "cls": rng.normal(size=(2, 5)),
    }
    ours = {k: v.copy() for k, v in ref.items()}
    ours["cls"][1] += 1e-3 * np.abs(ref["cls"][1]).max()  # above 1e-4 relative
    table = compare(ours, ref)
    assert table["passes"].tolist() == [True, False]


def test_fidelity_clips_one_per_id_snr_label():
    rows = [
        (t, i, snr, label, f"{k:04d}", "test")
        for t, i in [("fan", "id_00"), ("valve", "id_02")]
        for snr in (-6, 0, 6)
        for label in ("normal", "abnormal")
        for k in range(3)
    ]
    table = pd.DataFrame(
        rows, columns=["machine_type", "machine_id", "snr_db", "label", "clip", "split"]
    )
    drawn = fidelity_clips(table)
    assert len(drawn) == 2 * 3 * 2
    assert drawn.equals(fidelity_clips(table))
