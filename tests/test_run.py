import json

import numpy as np
import pandas as pd
import pytest
import torch

from noisemismatch.config import CONFIG_DIR, load_yaml
from noisemismatch.data.dataset import build_clip_table
from noisemismatch.models.base import AnomalyDetector, register
from noisemismatch.run import DONE, RunSpec, execute, resolve_condition, write_run

FRAMES = 64
STRICT = {"exclude_if_false": ["m6_argmax_agree"], "exclude_if_true": []}
PROVENANCE = {"git_commit": "abc123", "git_dirty": False}


@register
class _Projection(AnomalyDetector):
    """Scores = waveform projected on a random vector drawn at fit time."""

    name = "test_projection"
    training_scope = "machine_id"

    def fit(self, train_set):
        self.direction = torch.randn(FRAMES).double().numpy()  # global torch RNG

    def score(self, test_set):
        return test_set.waveforms() @ self.direction


class _Audio:
    fingerprint = "fake"

    def load(self, machine_type, snr, files):
        out = []
        for r in files.to_numpy():
            seed = abs(hash((machine_type, snr, *r))) % 2**32
            out.append(
                np.random.default_rng(seed).integers(-99, 99, FRAMES, dtype=np.int16)
            )
        return np.stack(out)


def _clip_table(types=(("fan", "id_00"), ("pump", "id_02"))):
    rows, split = [], []
    for machine_type, machine_id in types:
        for label, n in (("normal", 6), ("abnormal", 2)):
            for c in range(n):
                clip = f"{c:08d}"
                rows.append(
                    {
                        "machine_type": machine_type,
                        "machine_id": machine_id,
                        "label": label,
                        "clip": clip,
                        **{
                            f"{p}_{k}": v
                            for p in ("p6", "m6")
                            for k, v in (
                                ("machine_id", machine_id),
                                ("label", label),
                                ("clip", clip),
                            )
                        },
                        "kept": True,
                        "m6_argmax_agree": True,
                    }
                )
                in_test = label == "abnormal" or c < 2
                split.append(
                    (
                        machine_type,
                        machine_id,
                        label,
                        clip,
                        "test" if in_test else "train",
                    )
                )
    identities = pd.DataFrame(rows)
    split = pd.DataFrame(
        split, columns=["machine_type", "machine_id", "label", "clip", "split"]
    )
    return build_clip_table(split, identities, (0, 6, -6), STRICT)


def _run(table, seed=0, condition=("p6dB", (6,))):
    spec = RunSpec(
        "test_projection", "main", condition[0], condition[1], (0, 6, -6), seed
    )
    return execute(
        spec, table, _Audio(), None, {}, torch.device("cpu"), 0.1, PROVENANCE
    )


def test_scores_have_identity_provenance_and_one_row_per_clip_and_snr():
    table = _clip_table()
    scores, result = _run(table)
    n_test = (table["split"] == "test").sum()
    assert len(scores) == n_test  # the clip table already has one row per SNR
    for column in (
        "run_id",
        "seed",
        "git_commit",
        "machine_type",
        "machine_id",
        "label",
        "clip",
        "file_clip",
        "y",
        "strict_exclude",
        "score",
        "test_snr_db",
    ):
        assert column in scores.columns
    assert set(scores["git_commit"]) == {"abc123"}
    assert set(scores["seed"]) == {0}
    assert len(result["metrics"]) == 2 * 3  # 2 machine IDs x 3 test SNRs
    first = result["metrics"][0]
    assert first["n_test_normal"] == 2
    assert first["n_test_anomalous"] == 2
    assert 0.0 <= first["auc"] <= 1.0


def test_same_seed_same_scores_other_seed_other_scores():
    table = _clip_table()
    a, _ = _run(table, seed=1)
    b, _ = _run(table, seed=1)
    c, _ = _run(table, seed=2)
    assert a["score"].equals(b["score"])
    assert not a["score"].equals(c["score"])


def test_group_scores_do_not_depend_on_other_groups():
    both, _ = _run(_clip_table())
    alone, _ = _run(_clip_table(types=(("pump", "id_02"),)))
    pump = both[both["machine_type"] == "pump"].reset_index(drop=True)
    assert np.allclose(pump["score"], alone["score"])


def test_multi_condition_trains_on_every_snr():
    table = _clip_table()
    _, result = _run(table, condition=("multi", (0, 6, -6)))
    assert result["train_snrs_db"] == [0, 6, -6]


def test_resolve_condition_follows_the_protocol():
    protocol = load_yaml(CONFIG_DIR / "protocol.yaml")
    assert resolve_condition("-6", protocol, (0, 6, -6)) == ("m6dB", (-6,))
    assert resolve_condition("6", protocol, (0, 6)) == ("p6dB", (6,))
    assert resolve_condition("multi", protocol, (0, 6)) == ("multi", (0, 6))
    with pytest.raises(ValueError, match="not an SNR of this split"):
        resolve_condition("-6", protocol, (0, 6))


def test_write_run_writes_done_last(tmp_path):
    scores, result = _run(_clip_table())
    result["finished_utc"] = "2026-09-29T00:00:00+00:00"
    write_run(tmp_path, scores, result)
    assert (tmp_path / DONE).exists()
    assert json.loads((tmp_path / "result.json").read_text())["seed"] == 0
    reloaded = pd.read_csv(
        tmp_path / "scores.csv", dtype={"clip": str, "file_clip": str}
    )
    assert reloaded["clip"].iloc[0] == scores["clip"].iloc[0]
