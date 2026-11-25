"""Environment check: software versions, GPU access and CUDA determinism.

Usage::

    CUDA_VISIBLE_DEVICES=1 python -m noisemismatch.envcheck [--json out.json]

The check runs a tiny convolutional network forward and backward twice on the
first visible GPU (a few MB of memory, well under a second) and verifies that the
gradients are bitwise identical. Exit code 0 means CUDA is usable and deterministic.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch import nn

from noisemismatch.config import load_paths
from noisemismatch.utils.logs import setup_logging
from noisemismatch.utils.provenance import (
    environment_info,
    git_commit,
    git_is_dirty,
)
from noisemismatch.utils.repro import seed_everything

logger = logging.getLogger(__name__)


def _gradients_once(device: torch.device, seed: int) -> list[torch.Tensor]:
    """Runs one forward/backward pass of a small CNN and returns its gradients."""
    seed_everything(seed)
    model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 4),
    ).to(device)
    x = torch.randn(8, 1, 64, 64, device=device)
    target = torch.randint(0, 4, (8,), device=device)
    loss = nn.functional.cross_entropy(model(x), target)
    loss.backward()
    return [p.grad.detach().cpu().clone() for p in model.parameters()]


def check_determinism(device: torch.device, seed: int = 0) -> bool:
    """Tells whether two identical training steps give bitwise-identical gradients.

    Args:
        device: Device to run on.
        seed: Seed used for both runs.

    Returns:
        True if all gradients are bitwise identical.
    """
    first = _gradients_once(device, seed)
    second = _gradients_once(device, seed)
    return all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


def main(argv: list[str] | None = None) -> int:
    """Runs the environment check.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 if CUDA is available and deterministic, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, help="Also write the report to this file.")
    args = parser.parse_args(argv)
    setup_logging()

    # Seeding first sets CUBLAS_WORKSPACE_CONFIG before any CUDA work.
    seed_everything(0)
    report: dict[str, object] = {
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "environment": environment_info(),
    }
    paths = load_paths()
    report["storage_paths"] = {
        name: {"path": str(path), "exists": path.exists()}
        for name, path in vars(paths).items()
    }

    ok = torch.cuda.is_available()
    if ok:
        device = torch.device("cuda", 0)
        report["cuda_deterministic"] = check_determinism(device)
        report["cpu_deterministic"] = check_determinism(torch.device("cpu"))
        ok = bool(report["cuda_deterministic"]) and bool(report["cpu_deterministic"])
    else:
        logger.error("CUDA is not available: training requires a GPU.")

    text = json.dumps(report, indent=2)
    print(text)
    if args.json is not None:
        args.json.write_text(text + "\n", encoding="utf-8")
    logger.info("Environment check %s.", "passed" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
