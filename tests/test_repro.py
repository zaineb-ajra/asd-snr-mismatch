import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from noisemismatch.utils.repro import make_generator, seed_everything, seed_worker


def _draws() -> tuple[float, float, torch.Tensor]:
    return random.random(), np.random.rand(), torch.rand(3)  # noqa: NPY002


def test_seed_everything_repeats_all_generators():
    seed_everything(123)
    a = _draws()
    seed_everything(123)
    b = _draws()
    assert a[0] == b[0]
    assert a[1] == b[1]
    assert torch.equal(a[2], b[2])


def test_seed_everything_enables_deterministic_mode():
    seed_everything(0)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark


def _train_tiny_model(device: torch.device, seed: int) -> list[torch.Tensor]:
    seed_everything(seed)
    model = nn.Sequential(
        nn.Conv1d(1, 8, kernel_size=5, padding=2),
        nn.BatchNorm1d(8),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(8, 3),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.randn(32, 1, 100, device=device)
    y = torch.randint(0, 3, (32,), device=device)
    for _ in range(5):
        optimizer.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        optimizer.step()
    return [p.detach().cpu() for p in model.parameters()]


def _assert_same_weights(a: list[torch.Tensor], b: list[torch.Tensor]) -> None:
    assert all(torch.equal(p, q) for p, q in zip(a, b, strict=True))


def test_training_is_bitwise_deterministic_on_cpu():
    device = torch.device("cpu")
    _assert_same_weights(_train_tiny_model(device, 7), _train_tiny_model(device, 7))


def test_different_seeds_give_different_weights():
    device = torch.device("cpu")
    a, b = _train_tiny_model(device, 7), _train_tiny_model(device, 8)
    assert not all(torch.equal(p, q) for p, q in zip(a, b, strict=True))


@pytest.mark.gpu
def test_training_is_bitwise_deterministic_on_gpu():
    device = torch.device("cuda", 0)
    _assert_same_weights(_train_tiny_model(device, 7), _train_tiny_model(device, 7))


class _NumpyNoiseDataset(Dataset):
    """Returns values drawn from NumPy's global generator inside the worker."""

    def __len__(self) -> int:
        return 16

    def __getitem__(self, index: int) -> tuple[int, float]:
        return index, float(np.random.rand())  # noqa: NPY002


def _epoch(seed: int) -> list[tuple[int, float]]:
    loader = DataLoader(
        _NumpyNoiseDataset(),
        batch_size=4,
        shuffle=True,
        num_workers=2,
        # fork() in a multi-threaded process is unsafe (Python >= 3.12 warns).
        multiprocessing_context="spawn",
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
    )
    out: list[tuple[int, float]] = []
    for indices, values in loader:
        out.extend(zip(indices.tolist(), values.tolist(), strict=True))
    return out


def test_dataloader_order_and_worker_rng_are_reproducible():
    assert _epoch(3) == _epoch(3)
    assert _epoch(3) != _epoch(4)
