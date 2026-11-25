import importlib

import pytest

import noisemismatch
from noisemismatch.config import CONFIG_DIR, load_yaml

SUBMODULES = [
    "noisemismatch.config",
    "noisemismatch.envcheck",
    "noisemismatch.utils.logs",
    "noisemismatch.utils.provenance",
    "noisemismatch.utils.repro",
    "noisemismatch.data",
    "noisemismatch.data.download",
    "noisemismatch.data.mimii",
    "noisemismatch.data.pairing",
    "noisemismatch.data.prepare",
    "noisemismatch.data.split",
    "noisemismatch.data.matching",
    "noisemismatch.data.identity",
    "noisemismatch.data.build_split",
    "noisemismatch.data.cross_noise",
    "noisemismatch.data.audio_cache",
    "noisemismatch.data.dataset",
    "noisemismatch.features.base",
    "noisemismatch.features.logmel",
    "noisemismatch.features.cache",
    "noisemismatch.models.base",
    "noisemismatch.eval.metrics",
    "noisemismatch.eval.aggregate",
    "noisemismatch.eval.evaluate",
    "noisemismatch.run",
    "noisemismatch.features",
    "noisemismatch.models",
    "noisemismatch.eval",
]


def test_version_is_a_string():
    assert isinstance(noisemismatch.__version__, str)
    assert noisemismatch.__version__


@pytest.mark.parametrize("name", SUBMODULES)
def test_submodule_imports(name):
    importlib.import_module(name)


def test_committed_protocol_config():
    protocol = load_yaml(CONFIG_DIR / "protocol.yaml")
    snrs = protocol["dataset"]["snrs_db"]
    assert snrs == [-6, 0, 6]
    assert protocol["evaluation"]["test_snrs_db"] == snrs
    assert len(protocol["training"]["seeds"]) == 3
    conditions = {
        c["name"]: c["train_snrs_db"] for c in protocol["training"]["conditions"]
    }
    # One single-SNR condition per SNR, plus the multi-condition union.
    assert sorted(v[0] for k, v in conditions.items() if k != "multi") == snrs
    assert sorted(conditions["multi"]) == snrs
    assert protocol["evaluation"]["pauc_max_fpr"] == 0.1
    assert protocol["training"]["multi_condition_budget"] == "same_steps"
    aggregation = protocol["aggregation"]
    assert aggregation["over_ids"] == "arithmetic_mean"
    assert aggregation["over_machine_types"] == "arithmetic_mean"
    assert aggregation["seed_std_ddof"] == 1
    assert aggregation["robustness"] == [
        {
            "name": "fan_without_id_00",
            "drop": [{"machine_type": "fan", "machine_id": "id_00"}],
        }
    ]
    assert protocol["dataset"]["excluded_machine_ids"] == [
        {"machine_type": "valve", "machine_id": "id_00"}
    ]
    assert protocol["split"]["cluster_min_score_p6"] is None
