"""OS-SCL diagnostic on the DCASE 2020 Task 2 split (MIMII part).

Decided on 2026-10-05, after the study results (docs/validation.md, "OS-SCL:
diagnostic on the DCASE 2020 split"). The split is the one of the DCASE 2020
Task 2 development set for fan, pump, slider and valve, rebuilt from the MIMII
-6 dB archives with ``metadata/dcase2020_mimii_mapping.csv`` (no DCASE audio is
used). Arms, criterion and consequences: ``configs/dcase2020_check.yaml``.

Usage::

    python -m noisemismatch.dcase2020_check cache [--workers 12]
    python -m noisemismatch.dcase2020_check run --arm A --device cuda:0
    python -m noisemismatch.dcase2020_check report

``cache`` writes channel 0 of the 18,019 MIMII -6 dB files to its own audio cache
(``<cache_root>/audio/ch0-dcase2020/``). ``run`` trains on the DCASE ``train/``
clips and scores the ``test/`` clips; outputs go to
``<output_root>/runs/dcase2020/<model>/<arm>/seed<k>/`` (same files as a study
run). ``report`` writes ``results/dcase2020_check.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import torch

from noisemismatch.config import CONFIG_DIR, REPO_ROOT, load_paths, load_yaml
from noisemismatch.data.audio_cache import AudioCache, build_cache
from noisemismatch.data.dataset import ORDER
from noisemismatch.data.download import DOWNLOAD_SUBDIR, LOG_SUBDIR
from noisemismatch.data.split import ANOMALOUS, FILE_COLUMNS, NORMAL
from noisemismatch.run import DONE, RunSpec, execute, write_run
from noisemismatch.utils.logs import setup_logging
from noisemismatch.utils.provenance import environment_info, git_commit, git_is_dirty
from noisemismatch.utils.repro import seed_everything

logger = logging.getLogger(__name__)

CONFIG_PATH: Path = CONFIG_DIR / "dcase2020_check.yaml"
MAPPING_PATH: Path = REPO_ROOT / "metadata" / "dcase2020_mimii_mapping.csv"
REPORT_PATH: Path = REPO_ROOT / "results" / "dcase2020_check.md"
SPLIT_NAME: str = "dcase2020"
CACHE_NAME: str = "ch0-dcase2020"
TYPES: tuple[str, ...] = ("fan", "pump", "slider", "valve")


def clip_table(snr: int) -> pd.DataFrame:
    """Clip table of the DCASE 2020 split (one row per MIMII -6 dB file).

    Args:
        snr: SNR of the files (-6).

    Returns:
        Clip table with the columns of a study clip table; the identity is the
        MIMII file itself, ``split`` is the DCASE directory.
    """
    m = pd.read_csv(MAPPING_PATH, dtype=str)
    expected = {"normal": NORMAL, "anomaly": ANOMALOUS}
    if not (m["dcase_label"].map(expected) == m["mimii_label"]).all():
        raise ValueError("DCASE and MIMII labels disagree")
    if not (m["machine_id"] == m["mimii_machine_id"]).all():
        raise ValueError("DCASE and MIMII machine IDs disagree")
    table = pd.DataFrame(
        {
            "machine_type": m["machine_type"],
            "machine_id": m["mimii_machine_id"],
            "label": m["mimii_label"],
            "clip": m["mimii_clip"],
            "snr_db": snr,
            "file_machine_id": m["mimii_machine_id"],
            "file_label": m["mimii_label"],
            "file_clip": m["mimii_clip"],
            "split": m["dcase_split"],
            "strict_exclude": False,
        }
    )
    table["y"] = (table["label"] == ANOMALOUS).astype(int)
    return table.sort_values(ORDER, ignore_index=True)


def cache_root_dir(cache_root: Path) -> Path:
    """Directory of the audio cache of this diagnostic."""
    return cache_root / "audio" / CACHE_NAME


def build(workers: int) -> None:
    """Writes the channel-0 cache of the 18,019 MIMII -6 dB files."""
    paths = load_paths()
    config = load_yaml(CONFIG_PATH)
    files = clip_table(int(config["snr_db"]))[["machine_type", "snr_db", *FILE_COLUMNS]]
    manifest = build_cache(
        files,
        paths.data_root / DOWNLOAD_SUBDIR,
        cache_root_dir(paths.cache_root),
        workers,
    )
    logger.info("DCASE 2020 audio cache: %d arrays", len(manifest["entries"]))


def run(arm: str, device: str, force: bool, allow_dirty: bool) -> int:
    """Trains and scores one arm (one seed).

    Args:
        arm: ``A``, ``B`` or ``control``.
        device: Torch device.
        force: Rerun a finished run.
        allow_dirty: Allow uncommitted changes (recorded).

    Returns:
        Process exit code.
    """
    import noisemismatch.models  # noqa: F401  (registers the models)

    config = load_yaml(CONFIG_PATH)
    spec_arm = config["arms"][arm]
    snr, seed = int(config["snr_db"]), int(config["seed"])
    spec = RunSpec(spec_arm["model"], SPLIT_NAME, arm, (snr,), (snr,), seed)
    paths = load_paths()
    run_dir = paths.output_root / "runs" / SPLIT_NAME / spec.model / arm / f"seed{seed}"
    if (run_dir / DONE).exists() and not force:
        print(f"{run_dir}: already done, skipped")
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "run.log")
    dirty = git_is_dirty()
    if dirty and not allow_dirty:
        logger.error("Uncommitted changes: commit them or pass --allow-dirty")
        return 1
    provenance = {"git_commit": git_commit(), "git_dirty": dirty}
    model_config = {
        **load_yaml(CONFIG_DIR / "models" / f"{spec.model}.yaml"),
        **spec_arm["overrides"],
    }
    protocol = load_yaml(CONFIG_DIR / "protocol.yaml")
    seed_everything(seed)
    started = time.time()
    logger.info("Run %s, arm %s (commit %s)", run_dir, arm, provenance["git_commit"])
    scores, result = execute(
        spec,
        clip_table(snr),
        AudioCache(cache_root_dir(paths.cache_root)),
        paths.cache_root,
        model_config,
        torch.device(device),
        float(protocol["evaluation"]["pauc_max_fpr"]),
        provenance,
    )
    result.update(
        {
            "arm": arm,
            "diagnostic_config": config,
            "environment": environment_info(),
            "duration_s": time.time() - started,
            "finished_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        }
    )
    write_run(run_dir, scores, result)
    logger.info("Run %s done in %.1f s", run_dir, result["duration_s"])
    return 0


def summary(metrics: pd.DataFrame) -> dict[str, float]:
    """Mean AUC and pAUC per machine type (over its machines) and of the four types."""
    per_type = metrics.groupby("machine_type")[["auc", "pauc"]].mean()
    out = {
        f"{t}_{k}": float(per_type.loc[t, k]) for t in TYPES for k in ("auc", "pauc")
    }
    for k in ("auc", "pauc"):
        out[f"mean_{k}"] = float(per_type.loc[list(TYPES), k].mean())
    return out


def report() -> str:
    """Per-machine and per-type values of every finished arm, criterion and verdict."""
    config = load_yaml(CONFIG_PATH)
    paths = load_paths()
    target, tol = float(config["published_mean_auc"]), float(config["tolerance_auc"])
    lines = [
        "# OS-SCL diagnostic on the DCASE 2020 split (MIMII part)",
        "",
        "Decided on 2026-10-05, after the study results (docs/validation.md). "
        "Generated by `python -m noisemismatch.dcase2020_check report`. AUC and "
        "pAUC (%) per machine, mean of the machines of a type, mean of the four "
        f"types. Criterion: |mean AUC of the four types - {100 * target:.2f}| ≤ "
        f"{100 * tol:.1f} points.",
        "",
    ]
    passed: dict[str, bool] = {}
    for arm, spec in config["arms"].items():
        run_dir = (
            paths.output_root
            / "runs"
            / SPLIT_NAME
            / spec["model"]
            / arm
            / f"seed{config['seed']}"
        )
        if not (run_dir / DONE).exists():
            lines += [f"## Arm {arm} ({spec['model']}): not finished", ""]
            continue
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        metrics = pd.DataFrame(result["metrics"])
        s = summary(metrics)
        verdict = ""
        if arm != "control":
            passed[arm] = abs(s["mean_auc"] - target) <= tol
            verdict = " — criterion " + ("met" if passed[arm] else "not met")
        lines += [
            f"## Arm {arm} ({spec['model']}, overrides {spec['overrides'] or 'none'})",
            "",
            f"Mean of the four types: AUC {100 * s['mean_auc']:.2f}, pAUC "
            f"{100 * s['mean_pauc']:.2f}{verdict}. Commit {result['git_commit']}, "
            f"dirty {result['git_dirty']}, {result['duration_s'] / 3600:.2f} h.",
            "",
            "| Type | AUC | pAUC |",
            "|---|---|---|",
            *[
                f"| {t} | {100 * s[t + '_auc']:.2f} | {100 * s[t + '_pauc']:.2f} |"
                for t in TYPES
            ],
            "",
            "| Machine | normal | anomalous | AUC | pAUC |",
            "|---|---|---|---|---|",
            *[
                f"| {r.machine_type} {r.machine_id} | {r.n_test_normal} | "
                f"{r.n_test_anomalous} | {100 * r.auc:.2f} | {100 * r.pauc:.2f} |"
                for r in metrics.itertuples()
            ],
            "",
        ]
    if {"A", "B"} <= set(passed):
        if passed["A"]:
            outcome = "A met: implementation validated on this split."
        elif passed["B"]:
            outcome = (
                "A not met, B met: our OS-SCL is a deviating variant, removed from "
                "the counts and the conclusions."
            )
        else:
            outcome = "Neither met: the current state is kept."
        lines += [
            "## Consequence fixed before the runs",
            "",
            outcome + " The results of the main study do not change.",
            "",
        ]
    text = "\n".join(lines)
    REPORT_PATH.write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point (``cache``, ``run``, ``report``)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_cache = sub.add_parser("cache")
    p_cache.add_argument("--workers", type=int, default=12)
    p_run = sub.add_parser("run")
    p_run.add_argument("--arm", required=True, choices=["A", "B", "control"])
    p_run.add_argument("--device", default="cuda:0")
    p_run.add_argument("--force", action="store_true")
    p_run.add_argument("--allow-dirty", action="store_true")
    sub.add_parser("report")
    args = parser.parse_args(argv)
    if args.command == "cache":
        setup_logging(load_paths().data_root / LOG_SUBDIR / "audio_cache_dcase2020.log")
        build(args.workers)
        return 0
    if args.command == "run":
        return run(args.arm, args.device, args.force, args.allow_dirty)
    print(report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
