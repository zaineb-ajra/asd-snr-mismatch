"""Download of the pinned EAT checkpoint snapshot (Hugging Face, MIT License).

Usage::

    python -m noisemismatch.data.eat_checkpoint

downloads the files of ``configs/models/eat_knn.yaml`` (repository, revision) to
``<data_root>/eat/<name>/<revision>/`` and checks the SHA-256 of the weights. The
Python files of the snapshot are kept for the fidelity check only; the study code
uses the vendored encoder (``noisemismatch/features/eat_encoder.py``).
"""

from __future__ import annotations

import argparse
import logging
import urllib.request
from pathlib import Path

from noisemismatch.features.eat import checkpoint_dir, sha256

logger = logging.getLogger(__name__)

SNAPSHOT_FILES: tuple[str, ...] = (
    "README.md",
    "config.json",
    "configuration_eat.py",
    "eat_model.py",
    "model_core.py",
    "modeling_eat.py",
    "model.safetensors",
)


def download(data_root: Path, repo: str, revision: str, expected_sha256: str) -> Path:
    """Downloads the snapshot files (if absent) and checks the weights.

    Args:
        data_root: Root of the downloaded data.
        repo: Hugging Face repository.
        revision: Commit hash.
        expected_sha256: SHA-256 of ``model.safetensors``.

    Returns:
        The snapshot directory.
    """
    target = checkpoint_dir(data_root, repo, revision)
    target.mkdir(parents=True, exist_ok=True)
    for name in SNAPSHOT_FILES:
        path = target / name
        if not path.exists():
            url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
            logger.info("Downloading %s", url)
            urllib.request.urlretrieve(url, path)
    if sha256(target / "model.safetensors") != expected_sha256:
        raise ValueError("model.safetensors: SHA-256 differs from the configuration")
    logger.info("Snapshot verified in %s", target)
    return target


def main(argv: list[str] | None = None) -> int:
    """Downloads the checkpoint snapshot of ``configs/models/eat_knn.yaml``.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        0 on success.
    """
    from noisemismatch.config import CONFIG_DIR, load_paths, load_yaml

    argparse.ArgumentParser(description="Download the EAT checkpoint.").parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    c = load_yaml(CONFIG_DIR / "models" / "eat_knn.yaml")["checkpoint"]
    download(load_paths().data_root, c["repo"], c["revision"], c["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
