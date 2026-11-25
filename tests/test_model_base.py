import numpy as np
import pytest
import torch

from noisemismatch.models.base import (
    AnomalyDetector,
    batch_indices,
    build_model,
    register,
    training_steps,
)


class _Constant(AnomalyDetector):
    name = "test_constant"
    training_scope = "machine_id"

    def fit(self, train_set):
        self.fitted = len(train_set)

    def score(self, test_set):
        return np.zeros(len(test_set))


def test_registry_builds_and_rejects_duplicates():
    register(_Constant)
    model = build_model("test_constant", {"a": 1}, seed=3, device=torch.device("cpu"))
    assert isinstance(model, _Constant)
    assert model.seed == 3

    class Other(_Constant):
        name = "test_constant"

    with pytest.raises(ValueError, match="already registered"):
        register(Other)
    with pytest.raises(KeyError, match="unknown model"):
        build_model("nope", {}, 0, torch.device("cpu"))


def test_unknown_training_scope_is_rejected():
    class Bad(_Constant):
        name = "test_bad"
        training_scope = "per_clip"

    with pytest.raises(ValueError, match="training scope"):
        Bad({}, 0, torch.device("cpu"))


def test_training_steps():
    assert training_steps(1000, 64, 50, drop_last=False) == 50 * 16
    assert training_steps(1000, 64, 50, drop_last=True) == 50 * 15


@pytest.mark.parametrize("drop_last", [False, True])
def test_single_snr_budget_gives_the_original_epochs(drop_last):
    n, batch, epochs = 100, 32, 3
    steps = training_steps(n, batch, epochs, drop_last)
    batches = list(batch_indices(n, batch, steps, np.random.default_rng(0), drop_last))
    assert len(batches) == steps
    per_epoch = steps // epochs
    for e in range(epochs):
        seen = np.concatenate(batches[e * per_epoch : (e + 1) * per_epoch])
        # Each pass is a permutation (minus the dropped remainder).
        assert len(np.unique(seen)) == len(seen)
        assert len(seen) == (96 if drop_last else 100)


def test_multi_condition_gets_the_same_steps_on_three_times_more_data():
    steps = training_steps(100, 32, 3, drop_last=False)  # single-SNR budget: 12
    assert steps == 12
    batches = list(batch_indices(300, 32, steps, np.random.default_rng(0)))
    assert len(batches) == steps
    # One pass over the 300 examples takes 10 batches: 1.2 passes instead of 3.
    first_pass = np.concatenate(batches[:10])
    assert sorted(first_pass.tolist()) == list(range(300))
    assert sum(len(b) for b in batches) == 300 + 2 * 32


def test_batches_are_reproducible():
    a = list(batch_indices(50, 8, 20, np.random.default_rng(5)))
    b = list(batch_indices(50, 8, 20, np.random.default_rng(5)))
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
