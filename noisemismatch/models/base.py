"""Common interface of anomaly detectors, model registry and training budget."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, ClassVar

import numpy as np
import torch

from noisemismatch.data.dataset import SCOPES, ClipSet


class AnomalyDetector(ABC):
    """An anomalous sound detector trained on normal clips only.

    The runner creates one instance per training group of ``training_scope``
    (one per machine ID, per machine type, or a single one), calls :meth:`fit`
    with the group's training clips, then :meth:`score` with its test clips.

    Args:
        config: Model configuration (from ``configs/models/<name>.yaml``).
        seed: Seed of this run (the runner has already seeded all generators).
        device: Torch device.
    """

    name: ClassVar[str]
    training_scope: ClassVar[str]

    def __init__(self, config: dict[str, Any], seed: int, device: torch.device) -> None:
        if self.training_scope not in SCOPES:
            raise ValueError(f"unknown training scope {self.training_scope!r}")
        self.config = config
        self.seed = seed
        self.device = device

    @abstractmethod
    def fit(self, train_set: ClipSet) -> None:
        """Trains on normal clips (all SNR conditions of the run are included).

        Args:
            train_set: Training clips of one group.
        """

    @abstractmethod
    def score(self, test_set: ClipSet) -> np.ndarray:
        """Anomaly scores of the test clips (higher = more anomalous).

        Args:
            test_set: Test clips of one group, at one SNR.

        Returns:
            Float array shaped ``(len(test_set),)``.
        """


MODELS: dict[str, type[AnomalyDetector]] = {}


def register(cls: type[AnomalyDetector]) -> type[AnomalyDetector]:
    """Class decorator adding a detector to :data:`MODELS` under its ``name``."""
    if cls.name in MODELS and MODELS[cls.name] is not cls:
        raise ValueError(f"model name {cls.name!r} already registered")
    MODELS[cls.name] = cls
    return cls


def build_model(
    name: str, config: dict[str, Any], seed: int, device: torch.device
) -> AnomalyDetector:
    """Instantiates a registered detector.

    Args:
        name: Registered model name.
        config: Model configuration.
        seed: Run seed.
        device: Torch device.

    Returns:
        The detector.
    """
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(MODELS)}")
    return MODELS[name](config, seed, device)


def training_steps(
    n_examples: int, batch_size: int, epochs: int, drop_last: bool
) -> int:
    """Optimization steps of a single-SNR training run.

    The multi-condition model of the same method, machine group and seed gets the
    same number of steps (protocol, "Training budget of the multi-condition
    model"): pass the single-SNR example count here, not the union's.

    Args:
        n_examples: Training examples of one SNR condition.
        batch_size: Batch size.
        epochs: Epochs of the original method.
        drop_last: Whether incomplete last batches are dropped.

    Returns:
        Number of optimization steps.
    """
    per_epoch = (
        n_examples // batch_size if drop_last else math.ceil(n_examples / batch_size)
    )
    return epochs * per_epoch


def batch_indices(
    n_examples: int,
    batch_size: int,
    steps: int,
    generator: np.random.Generator,
    drop_last: bool = False,
) -> Iterator[np.ndarray]:
    """Yields ``steps`` batches of indices, reshuffling at each pass over the data.

    Batches never straddle two passes, so each pass is a permutation. With
    ``steps`` from :func:`training_steps` (same ``drop_last``), a single-SNR run
    makes exactly its original number of epochs, and a multi-condition run
    (three times more examples) about a third as many passes.

    Args:
        n_examples: Number of examples to draw from.
        batch_size: Batch size.
        steps: Number of batches to yield.
        generator: Random generator (seeded by the run).
        drop_last: Skip the incomplete last batch of each pass.

    Yields:
        Index arrays.
    """
    if n_examples < (batch_size if drop_last else 1):
        raise ValueError("not enough examples for one batch")
    done = 0
    while done < steps:
        order = generator.permutation(n_examples)
        for start in range(0, n_examples, batch_size):
            batch = order[start : start + batch_size]
            if drop_last and len(batch) < batch_size:
                break
            if done == steps:
                return
            yield batch
            done += 1
