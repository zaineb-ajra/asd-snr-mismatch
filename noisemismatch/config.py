"""Configuration loading: YAML files and machine-specific storage paths.

Storage locations (dataset, feature cache, run outputs) are never hard-coded. They
are read from ``configs/paths.yaml`` (committed defaults, relative to the repository
root) and optionally overridden, key by key, by a machine-specific file:

1. the YAML file named by the ``NOISEMISMATCH_PATHS`` environment variable, or
2. ``configs/paths.local.yaml`` (git-ignored), if it exists.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
CONFIG_DIR: Path = REPO_ROOT / "configs"
PATHS_ENV_VAR: str = "NOISEMISMATCH_PATHS"
LOCAL_PATHS_FILE: str = "paths.local.yaml"

_PATH_KEYS: tuple[str, ...] = ("data_root", "cache_root", "output_root")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Loads a YAML file whose top-level node is a mapping.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed mapping; an empty dict for an empty file.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the top-level YAML node is not a mapping.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: expected a mapping at top level, got {type(data).__name__}"
        )
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merges ``override`` into a copy of ``base``.

    Nested mappings are merged key by key; any other value in ``override``
    replaces the value in ``base``. Neither input is modified.

    Args:
        base: Default values.
        override: Values taking precedence over ``base``.

    Returns:
        A new merged dictionary.
    """
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(value, Mapping) and isinstance(current, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class StoragePaths:
    """Resolved, absolute storage locations.

    Attributes:
        data_root: Root of the MIMII dataset (downloaded archives and extracted audio).
        cache_root: Root of the feature cache.
        output_root: Root of per-run outputs (result files, per-clip scores, logs).
    """

    data_root: Path
    cache_root: Path
    output_root: Path


def load_paths(
    config_dir: Path = CONFIG_DIR,
    base_dir: Path = REPO_ROOT,
    env: Mapping[str, str] | None = None,
) -> StoragePaths:
    """Loads storage paths from the committed defaults and local overrides.

    Args:
        config_dir: Directory holding ``paths.yaml`` and the optional
            ``paths.local.yaml``.
        base_dir: Directory against which relative paths are resolved.
        env: Environment variables to read ``NOISEMISMATCH_PATHS`` from;
            defaults to ``os.environ``.

    Returns:
        The resolved storage paths. Directories are not created.

    Raises:
        KeyError: If a required key is missing after merging.
        ValueError: If an unknown key is present (typically a typo).
    """
    env = os.environ if env is None else env
    cfg = load_yaml(config_dir / "paths.yaml")
    override_file = env.get(PATHS_ENV_VAR)
    if override_file:
        cfg = deep_merge(cfg, load_yaml(override_file))
    elif (config_dir / LOCAL_PATHS_FILE).exists():
        cfg = deep_merge(cfg, load_yaml(config_dir / LOCAL_PATHS_FILE))

    missing = [key for key in _PATH_KEYS if key not in cfg]
    if missing:
        raise KeyError(f"Missing storage path keys: {missing}")
    unknown = sorted(set(cfg) - set(_PATH_KEYS))
    if unknown:
        raise ValueError(f"Unknown storage path keys: {unknown}")

    resolved: dict[str, Path] = {}
    for key in _PATH_KEYS:
        path = Path(os.path.expanduser(str(cfg[key])))
        resolved[key] = path if path.is_absolute() else (base_dir / path).resolve()
    return StoragePaths(**resolved)
