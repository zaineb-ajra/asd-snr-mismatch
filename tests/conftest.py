"""Shared pytest configuration: automatic skipping of hardware- and data-dependent tests."""

import pytest
import torch

from noisemismatch.data.prepare import index_path


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    skips = {}
    if not torch.cuda.is_available():
        skips["gpu"] = pytest.mark.skip(reason="CUDA not available")
    if not index_path().exists():
        skips["data"] = pytest.mark.skip(
            reason="MIMII file index not found (run noisemismatch.data.prepare)"
        )
    for item in items:
        for marker, skip in skips.items():
            if marker in item.keywords:
                item.add_marker(skip)
