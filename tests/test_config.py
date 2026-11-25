from pathlib import Path

import pytest

from noisemismatch.config import (
    CONFIG_DIR,
    PATHS_ENV_VAR,
    REPO_ROOT,
    deep_merge,
    load_paths,
    load_yaml,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def config_dir(tmp_path):
    d = tmp_path / "configs"
    d.mkdir()
    _write(
        d / "paths.yaml",
        "data_root: work/data\ncache_root: work/cache\noutput_root: work/outputs\n",
    )
    return d


def test_load_yaml_empty_file_is_empty_dict(tmp_path):
    assert load_yaml(_write(tmp_path / "e.yaml", "")) == {}


def test_load_yaml_rejects_non_mapping(tmp_path):
    with pytest.raises(ValueError, match="mapping"):
        load_yaml(_write(tmp_path / "l.yaml", "- a\n- b\n"))


def test_deep_merge_is_recursive_and_pure():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"d": 4}, "e": 5}
    assert deep_merge(base, override) == {"a": 1, "b": {"c": 2, "d": 4}, "e": 5}
    assert base == {"a": 1, "b": {"c": 2, "d": 3}}


def test_relative_defaults_resolve_against_base_dir(config_dir, tmp_path):
    paths = load_paths(config_dir, base_dir=tmp_path, env={})
    assert paths.data_root == (tmp_path / "work/data").resolve()
    assert paths.cache_root == (tmp_path / "work/cache").resolve()
    assert paths.output_root == (tmp_path / "work/outputs").resolve()


def test_local_file_overrides_defaults(config_dir, tmp_path):
    _write(config_dir / "paths.local.yaml", "data_root: /abs/data\n")
    paths = load_paths(config_dir, base_dir=tmp_path, env={})
    assert paths.data_root == Path("/abs/data")
    assert paths.cache_root == (tmp_path / "work/cache").resolve()


def test_env_file_takes_precedence_over_local_file(config_dir, tmp_path):
    _write(config_dir / "paths.local.yaml", "data_root: /from/local\n")
    env_file = _write(tmp_path / "env.yaml", "data_root: /from/env\n")
    paths = load_paths(
        config_dir, base_dir=tmp_path, env={PATHS_ENV_VAR: str(env_file)}
    )
    assert paths.data_root == Path("/from/env")


def test_unknown_key_is_rejected(config_dir, tmp_path):
    _write(config_dir / "paths.local.yaml", "data_rot: /typo\n")
    with pytest.raises(ValueError, match="data_rot"):
        load_paths(config_dir, base_dir=tmp_path, env={})


def test_missing_key_is_rejected(tmp_path):
    d = tmp_path / "configs"
    d.mkdir()
    _write(d / "paths.yaml", "data_root: x\n")
    with pytest.raises(KeyError, match="cache_root"):
        load_paths(d, base_dir=tmp_path, env={})


def test_committed_defaults_are_relative():
    # Committed defaults must not contain machine-specific absolute paths.
    defaults = load_yaml(CONFIG_DIR / "paths.yaml")
    assert all(not Path(str(v)).is_absolute() for v in defaults.values())
    paths = load_paths(CONFIG_DIR, base_dir=REPO_ROOT, env={})
    assert all(p.is_absolute() for p in vars(paths).values())
