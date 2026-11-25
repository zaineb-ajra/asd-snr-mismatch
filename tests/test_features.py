import json

import numpy as np
import pandas as pd
import pytest

from noisemismatch.features.cache import FeatureCache, cache_identity, cache_key
from noisemismatch.features.logmel import LogMel

FRAMES = 16000


class FakeAudio:
    """In-memory stand-in for the audio cache (one type, one SNR)."""

    def __init__(self, fingerprint="audio-v1", n=3, seed=0):
        self.fingerprint = fingerprint
        rng = np.random.default_rng(seed)
        self.data = rng.integers(-2000, 2000, size=(n, FRAMES), dtype=np.int16)
        self.files = pd.DataFrame(
            {
                "file_machine_id": ["id_00"] * n,
                "file_label": ["normal"] * n,
                "file_clip": [f"{i:08d}" for i in range(n)],
            }
        )
        self.array_calls = 0

    def index(self, machine_type, snr):
        return self.files

    def array(self, machine_type, snr):
        self.array_calls += 1
        return self.data


def test_logmel_shape_and_mimii_log_scale():
    extractor = LogMel()
    x = np.random.default_rng(0).uniform(-0.5, 0.5, 160_000).astype(np.float32)
    features = extractor(x)
    # 160,000 samples, hop 512, centered frames: 1 + 160000 // 512 = 313 frames.
    assert features.shape == (64, 313)
    assert features.dtype == np.float32
    silent = extractor(np.zeros(16000, dtype=np.float32))
    # 20 / power * log10(0 + eps) with power 2 and machine epsilon.
    assert np.allclose(silent, 10 * np.log10(np.finfo(float).eps))


def test_unknown_parameter_is_rejected():
    with pytest.raises(ValueError, match="unknown parameters"):
        LogMel(n_mel=64)


@pytest.mark.parametrize(
    "change",
    [
        {"n_mels": 128},
        {"hop_length": 256},
        {"pad_mode": "reflect"},
        {"power": 1.0},
        {"log_floor": "max"},
    ],
)
def test_every_parameter_changes_the_key(change):
    base = cache_key(cache_identity(LogMel(), "audio-v1"))
    assert cache_key(cache_identity(LogMel(**change), "audio-v1")) != base


def test_audio_fingerprint_changes_the_key():
    a = cache_key(cache_identity(LogMel(), "audio-v1"))
    assert cache_key(cache_identity(LogMel(), "audio-v2")) != a


def test_cache_matches_direct_extraction_and_is_reused(tmp_path):
    audio = FakeAudio()
    cache = FeatureCache(tmp_path, LogMel(), audio)
    request = audio.files.iloc[[2, 0]]
    features = cache.get("fan", 0, request)
    for row, k in zip(features, (2, 0), strict=True):
        assert np.array_equal(row, LogMel()(audio.data[k].astype(np.float32) / 32768))
    cache.get("fan", 0, request)
    FeatureCache(tmp_path, LogMel(), audio).get("fan", 0, request)
    assert audio.array_calls == 1  # computed once, then read from disk


def test_changed_parameters_never_read_a_stale_cache(tmp_path):
    audio = FakeAudio()
    old = FeatureCache(tmp_path, LogMel(n_mels=64), audio)
    old.get("fan", 0, audio.files)
    new = FeatureCache(tmp_path, LogMel(n_mels=32), audio)
    assert new.dir != old.dir
    assert new.get("fan", 0, audio.files).shape[1] == 32


def test_tampered_configuration_is_detected(tmp_path):
    audio = FakeAudio()
    cache = FeatureCache(tmp_path, LogMel(), audio)
    config = json.loads((cache.dir / "config.json").read_text())
    config["identity"]["extractor"]["config"]["n_mels"] = 128
    (cache.dir / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="stored configuration differs"):
        FeatureCache(tmp_path, LogMel(), audio)


def test_log_floor_modes_differ_only_near_zero():
    x = np.zeros(4000, dtype=np.float32)
    x[:2000] = np.random.default_rng(0).uniform(-0.5, 0.5, 2000)
    add, clip = LogMel()(x), LogMel(log_floor="max")(x)
    loud = add > -100  # frames with signal: mel >> eps, both modes agree
    assert np.allclose(add[loud], clip[loud], atol=1e-4)
    assert np.allclose(clip[~loud], 10 * np.log10(np.finfo(float).eps))
