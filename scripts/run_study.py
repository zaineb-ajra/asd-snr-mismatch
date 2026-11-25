"""Runs the study (Phase 6) as described in docs/phase6_checklist.md.

Usage (inside tmux, from a clean work tree)::

    python scripts/run_study.py --gpus 1 2

Stage A (autoencoder, MobileNetV2, EAT + kNN, Wilkinghoff; two processes per
GPU), then ``scripts/check_runs.py --stage A``; stage B (OS-SCL; one process per
GPU) starts only if every check passes. Finished runs are skipped, so the driver
can be restarted. Files written in ``<output_root>/runs``: ``PHASE6_COMMIT``,
``driver.log``, ``driver_failures.json``, ``logs/``, ``check_stage_<stage>.json``
and one of ``STAGE_A_PASSED`` / ``STAGE_A_FAILED``, then ``STAGE_B_PASSED`` /
``STAGE_B_FAILED``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_runs import STAGES, split_conditions

from noisemismatch.config import (
    CONFIG_DIR,
    REPO_ROOT,
    load_paths,
    load_yaml,
)
from noisemismatch.data.audio_cache import AudioCache, cache_dir
from noisemismatch.features.cache import FeatureCache
from noisemismatch.models.base import build_model
from noisemismatch.utils.logs import setup_logging
from noisemismatch.utils.provenance import git_commit, git_is_dirty

logger = logging.getLogger("run_study")

PROCESSES_PER_GPU: dict[str, int] = {"A": 2, "B": 1}
# Longest runs first, so that the queues end together.
ORDER: dict[str, tuple[str, ...]] = {
    "A": ("mobilenetv2", "ae", "wilkinghoff", "eat_knn"),
    "B": ("osscl",),
}
ATTEMPTS: int = 2
RETRY_PAUSE_S: int = 60
Job = tuple[str, str, str, str, int]  # model, split, condition, --train-snr, seed


def jobs_of(stage: str, protocol: dict) -> list[Job]:
    """Runs of a stage, in launch order."""
    assert set(ORDER[stage]) == set(STAGES[stage])
    return [
        (model, split, name, "multi" if len(snrs) > 1 else str(snrs[0]), seed)
        for model in ORDER[stage]
        for split, conditions in split_conditions(protocol).items()
        for name, snrs in conditions
        for seed in protocol["training"]["seeds"]
    ]


def warm_feature_caches(gpu: int, protocol: dict) -> None:
    """Computes the missing feature arrays, one at a time, before any run."""
    import noisemismatch.models  # noqa: F401  (registers the models)

    paths = load_paths()
    audio = AudioCache(cache_dir(paths.cache_root))
    device = torch.device(f"cuda:{gpu}")
    for name in ("ae", "mobilenetv2", "eat_knn"):
        config = load_yaml(CONFIG_DIR / "models" / f"{name}.yaml")
        model = build_model(name, config, 0, device)
        cache = FeatureCache(paths.cache_root, model.extractor, audio)
        computed = [
            (machine_type, snr)
            for machine_type in protocol["dataset"]["machine_types"]
            for snr in protocol["dataset"]["snrs_db"]
            if cache.ensure(machine_type, snr)
        ]
        logger.info(
            "Feature cache %s (%s): computed %s", cache.dir.name, name, computed
        )
        del model
    torch.cuda.empty_cache()


def worker(
    gpu: int,
    pending: queue.Queue[Job],
    runs: Path,
    failures: list[dict],
    lock: threading.Lock,
) -> None:
    """Runs jobs of the queue on one GPU until the queue is empty."""
    while True:
        try:
            model, split, condition, train_snr, seed = pending.get_nowait()
        except queue.Empty:
            return
        run_id = f"{split}/{model}/{condition}/seed{seed}"
        command = [
            "nice",
            "-n",
            "5",
            sys.executable,
            "-m",
            "noisemismatch.run",
            "--model",
            model,
            "--split",
            split,
            f"--train-snr={train_snr}",
            "--seed",
            str(seed),
            "--device",
            "cuda:0",
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        out = runs / "logs" / (run_id.replace("/", "_") + ".out")
        for attempt in range(1, ATTEMPTS + 1):
            started = time.time()
            with out.open("a", encoding="utf-8") as stream:
                code = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                ).returncode
            logger.info(
                "%s: attempt %d on GPU %d, exit code %d, %.0f s",
                run_id,
                attempt,
                gpu,
                code,
                time.time() - started,
            )
            if code == 0:
                break
            with lock:
                failures.append(
                    {
                        "run_id": run_id,
                        "attempt": attempt,
                        "gpu": gpu,
                        "exit_code": code,
                    }
                )
                (runs / "driver_failures.json").write_text(
                    json.dumps(failures, indent=2) + "\n"
                )
            if attempt < ATTEMPTS:
                time.sleep(RETRY_PAUSE_S)


def run_stage(
    stage: str, gpus: list[int], protocol: dict, runs: Path, failures: list[dict]
) -> bool:
    """Runs a stage and its checklist; returns True if every check passes."""
    jobs = jobs_of(stage, protocol)
    pending: queue.Queue[Job] = queue.Queue()
    for job in jobs:
        pending.put(job)
    logger.info(
        "Stage %s: %d runs, %d per GPU on GPUs %s",
        stage,
        len(jobs),
        PROCESSES_PER_GPU[stage],
        gpus,
    )
    lock = threading.Lock()
    threads = [
        threading.Thread(target=worker, args=(gpu, pending, runs, failures, lock))
        for gpu in gpus
        for _ in range(PROCESSES_PER_GPU[stage])
    ]
    started = time.time()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    logger.info(
        "Stage %s: runs finished in %.2f h", stage, (time.time() - started) / 3600
    )
    with (runs / f"check_stage_{stage}.out").open("w", encoding="utf-8") as stream:
        code = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "check_runs.py"),
                "--stage",
                stage,
            ],
            cwd=REPO_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    passed = code == 0
    (runs / f"STAGE_{stage}_{'PASSED' if passed else 'FAILED'}").write_text(
        time.strftime("%Y-%m-%d %H:%M:%S") + "\n"
    )
    logger.info("Stage %s: checklist %s", stage, "passed" if passed else "FAILED")
    return passed


def main(argv: list[str] | None = None) -> int:
    """Runs stage A, its checklist and, if it passes, stage B.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        0 if both stages pass their checklists.
    """
    parser = argparse.ArgumentParser(description="Driver of the study runs.")
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument(
        "--dry-run", action="store_true", help="List the runs and exit."
    )
    args = parser.parse_args(argv)
    protocol = load_yaml(CONFIG_DIR / "protocol.yaml")
    if args.dry_run:
        for stage in ("A", "B"):
            jobs = jobs_of(stage, protocol)
            print(f"stage {stage}: {len(jobs)} runs")
            for job in jobs:
                print("   ", *job)
        return 0
    runs = load_paths().output_root / "runs"
    (runs / "logs").mkdir(parents=True, exist_ok=True)
    setup_logging(runs / "driver.log")
    if git_is_dirty():
        logger.error("Uncommitted changes: the study runs from a clean work tree")
        return 1
    commit, marker = git_commit(), runs / "PHASE6_COMMIT"
    if marker.exists() and marker.read_text().strip() != commit:
        logger.error(
            "HEAD %s differs from PHASE6_COMMIT %s", commit, marker.read_text().strip()
        )
        return 1
    marker.write_text(commit + "\n")
    logger.info("Phase 6 at commit %s on GPUs %s", commit, args.gpus)
    warm_feature_caches(args.gpus[0], protocol)
    path = runs / "driver_failures.json"
    failures: list[dict] = json.loads(path.read_text()) if path.exists() else []
    if not run_stage("A", args.gpus, protocol, runs, failures):
        logger.error("Stage A failed its checklist: stage B is not started")
        return 1
    return 0 if run_stage("B", args.gpus, protocol, runs, failures) else 1


if __name__ == "__main__":
    raise SystemExit(main())
