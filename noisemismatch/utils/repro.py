"""Reproducibility helpers: global seeding and deterministic execution."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

# cuBLAS needs a fixed workspace configuration to be deterministic
# (https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility).
CUBLAS_WORKSPACE_CONFIG: str = ":4096:8"


def seed_everything(
    seed: int, deterministic: bool = True, warn_only: bool = False
) -> None:
    """Seeds Python, NumPy and PyTorch, and configures deterministic execution.

    Call this at program start, before any CUDA work: cuBLAS reads
    ``CUBLAS_WORKSPACE_CONFIG`` when its handle is created.

    Args:
        seed: Seed for all random number generators.
        deterministic: If True, disables cuDNN autotuning, forces deterministic
            cuDNN kernels and calls ``torch.use_deterministic_algorithms(True)``.
        warn_only: Forwarded to ``torch.use_deterministic_algorithms``. If True,
            operations without a deterministic implementation emit a warning
            instead of raising an error.
    """
    random.seed(seed)
    # Legacy global NumPy seeding is intentional: third-party code (e.g. librosa,
    # scikit-learn with random_state=None) draws from the global generator.
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)  # seeds the CPU and every CUDA device
    torch.backends.cudnn.benchmark = False
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    else:
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)


def seed_worker(worker_id: int) -> None:
    """Seeds NumPy and ``random`` inside a DataLoader worker.

    Use as ``DataLoader(worker_init_fn=seed_worker, generator=make_generator(seed))``.
    PyTorch already derives a distinct per-worker torch seed from the loader's
    generator; this propagates it to the other generators.

    Args:
        worker_id: Worker index (unused; the seed comes from ``torch.initial_seed``).
    """
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)  # noqa: NPY002
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Creates a seeded CPU ``torch.Generator`` (e.g. for DataLoader shuffling).

    Args:
        seed: Generator seed.

    Returns:
        The seeded generator.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
