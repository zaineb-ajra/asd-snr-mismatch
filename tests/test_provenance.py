import json
import logging
import subprocess

import pytest

from noisemismatch.utils.logs import setup_logging
from noisemismatch.utils.provenance import (
    environment_info,
    git_commit,
    git_is_dirty,
)


def _git(repo, *args):
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.org", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def test_git_functions_outside_a_repository(tmp_path):
    assert git_commit(tmp_path) is None
    assert git_is_dirty(tmp_path) is None


def test_git_state_unknown_before_first_commit(tmp_path):
    _git(tmp_path, "init", "-q")
    assert git_commit(tmp_path) is None
    assert git_is_dirty(tmp_path) is None


def test_git_commit_and_dirty_flag(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "f.txt").write_text("a")
    _git(tmp_path, "add", "f.txt")
    _git(tmp_path, "commit", "-q", "-m", "init")
    sha = git_commit(tmp_path)
    assert sha is not None
    assert len(sha) == 40
    assert git_is_dirty(tmp_path) is False
    (tmp_path / "untracked.txt").write_text("u")
    assert git_is_dirty(tmp_path) is False  # untracked files are ignored
    (tmp_path / "f.txt").write_text("b")
    assert git_is_dirty(tmp_path) is True


def test_environment_info_is_json_serializable():
    info = environment_info()
    json.dumps(info)
    assert info["packages"]["torch"] is not None
    assert "hostname" not in info


def test_setup_logging_writes_file_without_duplicates(tmp_path):
    log_file = tmp_path / "logs" / "run.log"
    setup_logging(log_file)
    setup_logging(log_file)  # a second call must not duplicate handlers
    logging.getLogger("test").info("hello")
    for handler in logging.getLogger().handlers:
        handler.flush()
    lines = [line for line in log_file.read_text().splitlines() if "hello" in line]
    assert len(lines) == 1


@pytest.fixture(autouse=True)
def _reset_logging():
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
