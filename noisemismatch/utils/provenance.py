"""Provenance information recorded with every result: git state and environment."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import torch

from noisemismatch.config import REPO_ROOT

# Packages whose exact versions are recorded with each result.
TRACKED_PACKAGES: tuple[str, ...] = (
    "noisemismatch",
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "librosa",
    "soundfile",
    "torch",
    "torchaudio",
    "torchvision",
)


def _git(args: list[str], repo_root: Path) -> str | None:
    """Runs a git command and returns its stripped stdout, or None on failure."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def git_commit(repo_root: Path = REPO_ROOT) -> str | None:
    """Returns the full SHA of ``HEAD``.

    Args:
        repo_root: Directory inside the git work tree.

    Returns:
        The commit SHA, or None if ``repo_root`` is not in a git repository or
        has no commit yet.
    """
    return _git(["rev-parse", "HEAD"], repo_root)


def git_is_dirty(repo_root: Path = REPO_ROOT) -> bool | None:
    """Tells whether tracked files differ from ``HEAD``.

    Untracked files are ignored: they cannot affect a run unless imported, and
    new source files must be committed before a run is considered reproducible.

    Args:
        repo_root: Directory inside the git work tree.

    Returns:
        True if tracked files have uncommitted changes, False if clean, None if
        git state cannot be determined (no repository, or no commit yet).
    """
    if git_commit(repo_root) is None:
        return None
    status = _git(["status", "--porcelain", "--untracked-files=no"], repo_root)
    if status is None:
        return None
    return status != ""


def _package_version(name: str) -> str | None:
    """Returns the installed version of a distribution, or None if absent."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def environment_info() -> dict[str, Any]:
    """Collects software and hardware information relevant to reproducibility.

    The host name is deliberately not recorded (results may be published).

    Returns:
        A JSON-serializable dictionary.
    """
    cuda_available = torch.cuda.is_available()
    gpus = (
        [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if cuda_available
        else []
    )
    cudnn_version = torch.backends.cudnn.version() if cuda_available else None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {name: _package_version(name) for name in TRACKED_PACKAGES},
        "torch_cuda": torch.version.cuda,
        "cudnn": cudnn_version,
        "cuda_available": cuda_available,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": gpus,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
