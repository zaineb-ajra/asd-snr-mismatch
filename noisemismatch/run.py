"""Single entry point of the experiments: one model, one training condition, one seed.

Usage::

    python -m noisemismatch.run --model ae --train-snr 6 --seed 0
    python -m noisemismatch.run --model ae --train-snr multi --seed 1 --split sensitivity

``--train-snr`` is -6, 0, 6 or ``multi`` (union of the split's SNRs). A run trains
one model per group of the model's training scope on the training clips of the
condition, then scores the test clips at every SNR of the split. Its directory
``<output_root>/runs/<split>/<model>/<condition>/seed<k>/`` receives:

- ``scores.csv``: one row per test clip and test SNR, with the clip identity, the
  file used, the label, the strict-variant flag, the run, seed and git commit;
- ``result.json``: full configuration, seed, git commit and state, environment,
  duration and AUC/pAUC per machine ID and test SNR;
- ``run.log``; ``DONE`` is written last. A run with ``DONE`` is skipped, so a
  launcher can be restarted at will; ``--force`` reruns it.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from noisemismatch.config import CONFIG_DIR, load_paths, load_yaml
from noisemismatch.data.audio_cache import AudioCache, cache_dir
from noisemismatch.data.build_split import SENSITIVITY_SPLIT_PATH, identity_sets
from noisemismatch.data.dataset import ClipSet, build_clip_table, select
from noisemismatch.data.identity import load_identity
from noisemismatch.data.prepare import SPLIT_PATH
from noisemismatch.data.split import (
    FILE_COLUMNS,
    IDENTITY_COLUMNS,
    TEST,
    TRAIN,
    load_split,
)
from noisemismatch.eval.evaluate import per_id_table
from noisemismatch.features.cache import AudioSource
from noisemismatch.models.base import MODELS, build_model
from noisemismatch.utils.logs import setup_logging
from noisemismatch.utils.provenance import environment_info, git_commit, git_is_dirty
from noisemismatch.utils.repro import seed_everything

logger = logging.getLogger(__name__)

SPLIT_FILES: dict[str, Path] = {
    "main": SPLIT_PATH,
    "sensitivity": SENSITIVITY_SPLIT_PATH,
}
DONE: str = "DONE"


@dataclass(frozen=True)
class RunSpec:
    """What a run does.

    Attributes:
        model: Registered model name.
        split_name: ``main`` or ``sensitivity``.
        condition: Training condition name (``m6dB``, ``0dB``, ``p6dB``, ``multi``).
        train_snrs: SNRs of the training clips.
        test_snrs: SNRs of the test clips.
        seed: Seed.
    """

    model: str
    split_name: str
    condition: str
    train_snrs: tuple[int, ...]
    test_snrs: tuple[int, ...]
    seed: int

    @property
    def run_id(self) -> str:
        """Unique name of the run."""
        return f"{self.split_name}/{self.model}/{self.condition}/seed{self.seed}"


def resolve_condition(
    train_snr: str, protocol: dict, split_snrs: tuple[int, ...]
) -> tuple[str, tuple[int, ...]]:
    """Maps ``--train-snr`` to a condition of the protocol.

    Args:
        train_snr: ``-6``, ``0``, ``6`` or ``multi``.
        protocol: Parsed protocol.
        split_snrs: SNRs of the split.

    Returns:
        ``(condition name, training SNRs)``; ``multi`` uses the split's SNRs.
    """
    if train_snr == "multi":
        return "multi", tuple(split_snrs)
    snr = int(train_snr)
    if snr not in split_snrs:
        raise ValueError(f"training SNR {snr} is not an SNR of this split {split_snrs}")
    for condition in protocol["training"]["conditions"]:
        if condition["train_snrs_db"] == [snr]:
            return condition["name"], (snr,)
    raise ValueError(f"no protocol condition trains at {snr} dB")


def execute(
    spec: RunSpec,
    clip_table: pd.DataFrame,
    audio: AudioSource,
    cache_root: Path | None,
    model_config: dict[str, Any],
    device: torch.device,
    max_fpr: float,
    provenance: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Trains and scores one run (no file is written).

    Args:
        spec: Run specification.
        clip_table: Clip table of the split.
        audio: Audio cache.
        cache_root: Root of the feature caches.
        model_config: Model configuration.
        device: Torch device.
        max_fpr: pAUC bound.
        provenance: Git commit and state, recorded in every score row.

    Returns:
        ``(scores, result)``: per-clip scores and the result record (without
        environment and timing, added by the caller).
    """
    scope = MODELS[spec.model].training_scope
    train = ClipSet(select(clip_table, TRAIN, spec.train_snrs), audio, cache_root)
    test = ClipSet(select(clip_table, TEST, spec.test_snrs), audio, cache_root)
    test_groups = dict(test.groups(scope))
    parts = []
    diagnostics: dict[str, Any] = {}
    for key, train_group in train.groups(scope):
        seed_everything(spec.seed)  # independent of the group order
        model = build_model(spec.model, model_config, spec.seed, device)
        logger.info(
            "%s: fitting group %s on %d clips",
            spec.run_id,
            key or "all",
            len(train_group),
        )
        model.fit(train_group)
        if getattr(model, "diagnostics", None):
            diagnostics["/".join(key) or "all"] = model.diagnostics
        for snr in spec.test_snrs:
            group = test_groups[key].subset(test_groups[key].table["snr_db"] == snr)
            values = np.asarray(model.score(group), dtype=float)
            if values.shape != (len(group),):
                raise ValueError(f"{spec.model}.score returned shape {values.shape}")
            rows = group.table[
                [*IDENTITY_COLUMNS, *FILE_COLUMNS, "y", "strict_exclude"]
            ].copy()
            rows.insert(0, "test_snr_db", snr)
            rows["score"] = values
            parts.append(rows)
    scores = pd.concat(parts, ignore_index=True)
    for column, value in reversed(
        [
            ("run_id", spec.run_id),
            ("model", spec.model),
            ("split_name", spec.split_name),
            ("train_condition", spec.condition),
            ("seed", spec.seed),
            ("git_commit", provenance["git_commit"]),
        ]
    ):
        scores.insert(0, column, value)
    result = {
        "run_id": spec.run_id,
        "model": spec.model,
        "split_name": spec.split_name,
        "train_condition": spec.condition,
        "train_snrs_db": list(spec.train_snrs),
        "test_snrs_db": list(spec.test_snrs),
        "seed": spec.seed,
        "training_scope": scope,
        "model_config": model_config,
        **provenance,
        "metrics": per_id_table(scores, max_fpr).to_dict("records"),
        "model_diagnostics": diagnostics,
    }
    return scores, result


def write_run(run_dir: Path, scores: pd.DataFrame, result: dict[str, Any]) -> None:
    """Writes the outputs of a run, ``DONE`` last."""
    scores.to_csv(
        run_dir / "scores.csv", index=False, lineterminator="\n", float_format="%.8g"
    )
    (run_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / DONE).write_text(result["finished_utc"] + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Runs one (model, training condition, seed).

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code (0 on success or when the run is already done).
    """
    parser = argparse.ArgumentParser(description="Run one experiment.")
    parser.add_argument("--model", required=True, help="Registered model name.")
    parser.add_argument("--train-snr", required=True, help="-6, 0, 6 or multi.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", default="main", choices=sorted(SPLIT_FILES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true", help="Rerun a finished run.")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow uncommitted changes (recorded in the result).",
    )
    args = parser.parse_args(argv)

    import noisemismatch.models  # noqa: F401  (registers the models)

    paths = load_paths()
    protocol = load_yaml(CONFIG_DIR / "protocol.yaml")
    split_snrs = (
        tuple(protocol["evaluation"]["test_snrs_db"])
        if args.split == "main"
        else tuple(protocol["sensitivity"]["snrs_db"])
    )
    condition, train_snrs = resolve_condition(args.train_snr, protocol, split_snrs)
    spec = RunSpec(args.model, args.split, condition, train_snrs, split_snrs, args.seed)
    run_dir = paths.output_root / "runs" / spec.run_id
    if (run_dir / DONE).exists() and not args.force:
        print(f"{spec.run_id}: already done, skipped")
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "run.log")

    dirty = git_is_dirty()
    if dirty and not args.allow_dirty:
        logger.error("Uncommitted changes: commit them or pass --allow-dirty")
        return 1
    provenance = {"git_commit": git_commit(), "git_dirty": dirty}
    seed_everything(spec.seed)
    model_config = load_yaml(CONFIG_DIR / "models" / f"{spec.model}.yaml")
    identity = load_identity()
    sets = identity_sets(identity, protocol["dataset"]["excluded_machine_ids"])
    clip_table = build_clip_table(
        load_split(SPLIT_FILES[spec.split_name]),
        sets[spec.split_name],
        split_snrs,
        protocol["evaluation"]["strict_variant"],
    )
    audio = AudioCache(cache_dir(paths.cache_root))

    started = time.time()
    logger.info(
        "Run %s (commit %s, dirty=%s)", spec.run_id, provenance["git_commit"], dirty
    )
    scores, result = execute(
        spec,
        clip_table,
        audio,
        paths.cache_root,
        model_config,
        torch.device(args.device),
        float(protocol["evaluation"]["pauc_max_fpr"]),
        provenance,
    )
    result.update(
        {
            "protocol": protocol,
            "environment": environment_info(),
            "duration_s": time.time() - started,
            "finished_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        }
    )
    write_run(run_dir, scores, result)
    logger.info("Run %s done in %.1f s", spec.run_id, result["duration_s"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
