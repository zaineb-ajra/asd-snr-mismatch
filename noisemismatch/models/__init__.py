"""Anomalous sound detection models behind a common ``fit`` / ``score`` interface."""

from noisemismatch.models import (  # noqa: F401  (registration)
    ae,
    eat_knn,
    mobilenetv2,
    osscl,
    wilkinghoff,
)
from noisemismatch.models.base import MODELS, AnomalyDetector, build_model, register

__all__ = ["MODELS", "AnomalyDetector", "build_model", "register"]
