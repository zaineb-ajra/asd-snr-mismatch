"""Checks of the versioned identity table and split, and of both against the data.

Tests without the ``data`` marker read only versioned files under ``metadata/``.
The ``data`` tests need the MIMII archives and file index on disk.
"""

import numpy as np
import pytest

from noisemismatch.config import CONFIG_DIR, load_paths, load_yaml
from noisemismatch.data.build_split import SENSITIVITY_SPLIT_PATH, identity_sets
from noisemismatch.data.download import DOWNLOAD_SUBDIR
from noisemismatch.data.identity import IDENTITY_PATH, load_identity
from noisemismatch.data.matching import standardize
from noisemismatch.data.mimii import read_member
from noisemismatch.data.prepare import SPLIT_PATH
from noisemismatch.data.split import (
    IDENTITY_COLUMNS,
    assign_files,
    check_content_identity,
    check_no_leakage,
    identity_files,
    load_split,
    make_split,
)

needs_identity = pytest.mark.skipif(
    not IDENTITY_PATH.exists(), reason="identity table not generated yet"
)
needs_split = pytest.mark.skipif(
    not (SPLIT_PATH.exists() and SENSITIVITY_SPLIT_PATH.exists()),
    reason="splits not generated yet",
)


@pytest.fixture(scope="module")
def identity():
    return load_identity()


@pytest.fixture(scope="module")
def protocol():
    return load_yaml(CONFIG_DIR / "protocol.yaml")


@pytest.fixture(scope="module")
def sets(identity, protocol):
    return identity_sets(identity, protocol["dataset"]["excluded_machine_ids"])


@pytest.fixture(scope="module")
def rule(protocol):
    return protocol["identity"]


@needs_identity
def test_kept_identities_are_matched_one_to_one(identity):
    kept = identity[identity["kept"]]
    assert (kept["p6_accepted"] & kept["m6_accepted"]).all()
    for prefix in ("p6", "m6"):
        keys = [
            "machine_type",
            *(f"{prefix}_{k}" for k in ("machine_id", "label", "clip")),
        ]
        assert (kept[keys[1:]] != "").all().all()
        assert not kept.duplicated(keys).any()
    check_content_identity(identity_files(identity))


@needs_identity
def test_accepted_scores_satisfy_the_rule(identity, rule):
    kept = identity[identity["kept"]]
    for prefix in ("p6", "m6"):
        assert (kept[f"{prefix}_score"].astype(float) >= rule["min_score"]).all()
        assert (kept[f"{prefix}_margin"].astype(float) >= rule["min_margin"]).all()
    for term in ("m6_term0", "m6_term6"):
        assert (kept[term].astype(float) >= rule["min_support_m6"]).all()


@needs_identity
@needs_split
@pytest.mark.parametrize(
    ("name", "path", "snrs"),
    [("main", SPLIT_PATH, (0, 6, -6)), ("sensitivity", SENSITIVITY_SPLIT_PATH, (0, 6))],
)
def test_split_covers_its_identities_without_leakage(sets, name, path, snrs):
    split = load_split(path)  # MIMII protocol invariants, exact balance
    key = list(IDENTITY_COLUMNS)
    expected = sets[name][key]
    assert sorted(map(tuple, split[key].values)) == sorted(map(tuple, expected.values))
    excluded = (split["machine_type"] == "valve") & (split["machine_id"] == "id_00")
    assert not excluded.any()
    files = assign_files(split, identity_files(sets[name], snrs=snrs, only_kept=False))
    check_content_identity(files)
    check_no_leakage(files)
    assert sorted(files["snr_db"].unique()) == sorted(snrs)


@needs_identity
@needs_split
@pytest.mark.parametrize(
    ("name", "path"), [("main", SPLIT_PATH), ("sensitivity", SENSITIVITY_SPLIT_PATH)]
)
def test_committed_splits_are_reproducible(sets, protocol, name, path):
    seed = protocol["split"]["seed"]
    regenerated = make_split(sets[name][list(IDENTITY_COLUMNS)], seed)
    regenerated = regenerated.astype(str).reset_index(drop=True)
    assert regenerated.equals(load_split(path).reset_index(drop=True))


def _score(a: np.ndarray, b: np.ndarray) -> float:
    za, zb = standardize(a), standardize(b)
    return float((za * zb).sum() / za.size)


@pytest.mark.data
@needs_identity
def test_stored_scores_match_the_audio(identity):
    """Recomputes, from the archives, the scores of 12 random kept identities."""
    downloads = load_paths().data_root / DOWNLOAD_SUBDIR
    kept = identity[identity["kept"]].sample(12, random_state=0)

    def load(snr, machine_type, machine_id, label, clip):
        archive = downloads / f"{snr}_dB_{machine_type}.zip"
        member = f"{machine_type}/{machine_id}/{label}/{clip}.wav"
        return read_member(archive, member)[0]

    for r in kept.itertuples():
        x0 = load(0, r.machine_type, r.machine_id, r.label, r.clip)
        x6 = load(6, r.machine_type, r.p6_machine_id, r.p6_label, r.p6_clip)
        xm = load(-6, r.machine_type, r.m6_machine_id, r.m6_label, r.m6_clip)
        assert _score(x0, x6) == pytest.approx(float(r.p6_score), abs=2e-3)
        assert _score(x0, xm) == pytest.approx(float(r.m6_term0), abs=2e-3)
        assert _score(x6, xm) == pytest.approx(float(r.m6_term6), abs=2e-3)
