"""EAT frozen encoder + kNN (pre-trained-encoder family).

The encoder (EAT-base, epoch 30, pre-trained only; Chen et al., IJCAI 2024) is not
trained. A clip is embedded once (``noisemismatch.features.eat``, cached); its
anomaly score is the mean cosine distance to its K nearest embeddings in the
reference set, the normal training clips of its machine ID. The configuration
(temporal-mean pooling, K = 1, cosine) was fixed before any embedding was
computed (docs/protocol.md, "EAT + kNN configuration"); the scoring variants of
the sensitivity analysis are available through :meth:`EatKnnDetector.score_variants`.

The pinned checkpoint snapshot is downloaded by
``python -m noisemismatch.data.eat_checkpoint``.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np
import torch

from noisemismatch.data.dataset import ClipSet
from noisemismatch.features.eat import POOLINGS, EatEmbedding
from noisemismatch.models.base import AnomalyDetector, register

logger = logging.getLogger(__name__)

VARIANTS: tuple[tuple[str, int], ...] = (
    ("time_mean", 1),
    ("time_mean", 5),
    ("cls", 1),
    ("cls", 5),
)


def knn_scores(
    reference: np.ndarray, queries: np.ndarray, k: int, device: torch.device
) -> np.ndarray:
    """Mean cosine distance of each query to its ``k`` nearest reference rows.

    Args:
        reference: Array ``(n_reference, dim)``.
        queries: Array ``(n_queries, dim)``.
        k: Number of neighbours.
        device: Torch device.

    Returns:
        Float array ``(n_queries,)``.
    """
    if k > len(reference):
        raise ValueError(f"k = {k} exceeds the {len(reference)} reference embeddings")
    ref = torch.nn.functional.normalize(
        torch.from_numpy(reference).to(device, torch.float64), dim=1
    )
    qry = torch.nn.functional.normalize(
        torch.from_numpy(queries).to(device, torch.float64), dim=1
    )
    distances = 1.0 - qry @ ref.T
    nearest = torch.topk(distances, k, dim=1, largest=False).values
    return nearest.mean(dim=1).cpu().numpy()


@register
class EatKnnDetector(AnomalyDetector):
    """EAT embeddings + kNN (see module docstring)."""

    name: ClassVar[str] = "eat_knn"
    training_scope: ClassVar[str] = "machine_id"

    def __init__(self, config: dict[str, Any], seed: int, device: torch.device) -> None:
        super().__init__(config, seed, device)
        if config["metric"] != "cosine":
            raise ValueError(f"unsupported metric {config['metric']!r}")
        self.extractor = EatEmbedding(
            device=device, **config["checkpoint"], **config["features"]
        )
        self.reference: np.ndarray | None = None

    def fit(self, train_set: ClipSet) -> None:
        """Stores the embeddings of the normal training clips (no training).

        Args:
            train_set: Normal training clips of one machine ID.
        """
        self.reference = train_set.features(self.extractor)

    def score_variants(
        self, test_set: ClipSet, variants: tuple[tuple[str, int], ...] = VARIANTS
    ) -> dict[tuple[str, int], np.ndarray]:
        """Anomaly scores for several (pooling, K) variants.

        Args:
            test_set: Test clips of the machine ID.
            variants: ``(pooling, k)`` pairs; pooling in :data:`POOLINGS`.

        Returns:
            Scores per variant.
        """
        if self.reference is None:
            raise RuntimeError("fit must be called before scoring")
        queries = test_set.features(self.extractor)
        return {
            (pooling, k): knn_scores(
                self.reference[:, POOLINGS[pooling]],
                queries[:, POOLINGS[pooling]],
                k,
                self.device,
            )
            for pooling, k in variants
        }

    def score(self, test_set: ClipSet) -> np.ndarray:
        """Anomaly scores of the main configuration.

        Args:
            test_set: Test clips of the machine ID.

        Returns:
            Float array shaped ``(len(test_set),)``.
        """
        main = (self.config["pooling"], int(self.config["k"]))
        return self.score_variants(test_set, (main,))[main].astype(float)
