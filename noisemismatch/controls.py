"""Internal controls of a model with an auxiliary ID-classification task.

On the study data (main split), trains the model for one training condition and
seed, then records, per training group: the training curves; the accuracy and
confusion matrix of the auxiliary machine-ID classification on the normal test
clips at every test SNR (a class systematically confused would point to a
labelling problem rather than a weak model); the distributions of the anomaly
scores of normal and anomalous clips and the AUC per machine ID; and a
determinism check (the smallest group retrained with the same seed must give
identical scores). These runs check the implementation; they are not results of
the study.

Usage::

    CUDA_VISIBLE_DEVICES=2 python -m noisemismatch.controls --model mobilenetv2 --train-snr 0

Outputs: ``<output_root>/verification/<model>/internal/<condition>/seed<k>/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from noisemismatch.config import CONFIG_DIR, load_paths, load_yaml
from noisemismatch.data.audio_cache import AudioCache, cache_dir
from noisemismatch.data.build_split import identity_sets
from noisemismatch.data.dataset import ClipSet, build_clip_table, select
from noisemismatch.data.identity import load_identity
from noisemismatch.data.prepare import SPLIT_PATH
from noisemismatch.data.split import NORMAL, TEST, TRAIN, load_split
from noisemismatch.eval.metrics import auc
from noisemismatch.models.base import MODELS, AnomalyDetector, build_model
from noisemismatch.run import resolve_condition
from noisemismatch.utils.logs import setup_logging
from noisemismatch.utils.provenance import environment_info, git_commit, git_is_dirty
from noisemismatch.utils.repro import seed_everything

logger = logging.getLogger(__name__)


def id_confusion(model: Any, clips: ClipSet) -> pd.DataFrame:
    """Confusion matrix of the auxiliary ID task (rows true, columns predicted).

    Args:
        model: Trained model exposing ``classes`` and ``id_probabilities``.
        clips: Clips of known machine ID.

    Returns:
        Count table over the model's classes.
    """
    predicted = model.id_probabilities(clips).mean(axis=1).argmax(axis=1)
    true = (
        model.class_labels(clips.table)
        if hasattr(model, "class_labels")
        else clips.table["machine_id"]
    )
    return pd.crosstab(
        pd.Categorical(true, categories=model.classes),
        pd.Categorical([model.classes[k] for k in predicted], categories=model.classes),
        rownames=["true"],
        colnames=["predicted"],
        dropna=False,
    )


def score_summary(clips: ClipSet, scores: np.ndarray) -> list[dict[str, Any]]:
    """Quartiles of the scores of normal and anomalous clips and AUC, per ID."""
    out = []
    table = clips.table.assign(score=scores)
    for (machine_type, machine_id), part in table.groupby(
        ["machine_type", "machine_id"]
    ):
        record: dict[str, Any] = {
            "machine_type": machine_type,
            "machine_id": machine_id,
        }
        for label, values in part.groupby("label")["score"]:
            record[f"{label}_quartiles"] = np.quantile(
                values, [0.25, 0.5, 0.75]
            ).tolist()
        record["auc"] = auc(part["y"].to_numpy(), part["score"].to_numpy())
        out.append(record)
    return out


def _train(
    name: str, config: dict, seed: int, device: torch.device, train: ClipSet
) -> AnomalyDetector:
    seed_everything(seed)
    model = build_model(name, config, seed, device)
    model.fit(train)
    return model


def main(argv: list[str] | None = None) -> int:
    """Runs the internal controls of one model.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        0 if the determinism check passes.
    """
    parser = argparse.ArgumentParser(description="Internal controls of a model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-snr", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--machine-types", nargs="+", default=None)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Smoke test: override the epochs (outputs go to internal-smoke/).",
    )
    parser.add_argument(
        "--determinism-epochs",
        type=int,
        default=None,
        help="Check determinism with two trainings of this many epochs "
        "(default: retrain the smallest group fully).",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)

    import noisemismatch.models  # noqa: F401  (registers the models)

    paths = load_paths()
    protocol = load_yaml(CONFIG_DIR / "protocol.yaml")
    snrs = tuple(protocol["evaluation"]["test_snrs_db"])
    condition, train_snrs = resolve_condition(args.train_snr, protocol, snrs)
    out = (
        paths.output_root
        / "verification"
        / args.model
        / ("internal" if args.epochs is None else "internal-smoke")
        / condition
        / f"seed{args.seed}"
    )
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / "controls.log")
    dirty = git_is_dirty()
    if dirty and not args.allow_dirty:
        logger.error("Uncommitted changes: commit them or pass --allow-dirty")
        return 1
    config = load_yaml(CONFIG_DIR / "models" / f"{args.model}.yaml")
    if args.epochs is not None:
        config["epochs"] = args.epochs
    device = torch.device(args.device)
    sets = identity_sets(load_identity(), protocol["dataset"]["excluded_machine_ids"])
    table = build_clip_table(
        load_split(SPLIT_PATH),
        sets["main"],
        snrs,
        protocol["evaluation"]["strict_variant"],
    )
    audio = AudioCache(cache_dir(paths.cache_root))
    types = None if args.machine_types is None else tuple(args.machine_types)
    train = ClipSet(select(table, TRAIN, train_snrs, types), audio, paths.cache_root)
    test = ClipSet(select(table, TEST, snrs, types), audio, paths.cache_root)
    scope = MODELS[args.model].training_scope
    test_groups = dict(test.groups(scope))

    groups: dict[str, Any] = {}
    smallest: tuple[int, tuple, ClipSet, np.ndarray] | None = None
    for key, train_group in train.groups(scope):
        started = time.time()
        model = _train(args.model, config, args.seed, device, train_group)
        record: dict[str, Any] = {
            "history": model.history,
            "diagnostics": getattr(model, "diagnostics", None),
            "per_snr": {},
        }
        if hasattr(model, "save_weights"):
            model.save_weights(out / f"weights_{'_'.join(key) or 'all'}.pt")
        for snr in snrs:
            group = test_groups[key].subset(test_groups[key].table["snr_db"] == snr)
            normal = group.subset(group.table["label"] == NORMAL)
            matrix = id_confusion(model, normal)
            scores = model.score(group)
            record["per_snr"][str(snr)] = {
                "id_accuracy": float(
                    np.trace(matrix.to_numpy()) / matrix.to_numpy().sum()
                ),
                "confusion": {str(k): v for k, v in matrix.to_dict("index").items()},
                "scores": score_summary(group, scores),
            }
            if snr == train_snrs[0] and (
                smallest is None or len(train_group) < smallest[0]
            ):
                smallest = (len(train_group), key, group, scores)
        record["duration_s"] = time.time() - started
        groups["/".join(key) or "all"] = record
        logger.info(
            "%s: ID accuracy per test SNR %s (%.0f s)",
            key,
            {s: round(v["id_accuracy"], 4) for s, v in record["per_snr"].items()},
            record["duration_s"],
        )

    assert smallest is not None
    _, key, group, scores = smallest
    train_group = dict(train.groups(scope))[key]
    if args.determinism_epochs is None:
        again = _train(args.model, config, args.seed, device, train_group)
        deterministic = bool(np.array_equal(scores, again.score(group)))
    else:  # two short trainings instead of retraining the whole model
        short = {**config, "epochs": args.determinism_epochs}
        first = _train(args.model, short, args.seed, device, train_group).score(group)
        second = _train(args.model, short, args.seed, device, train_group).score(group)
        deterministic = bool(np.array_equal(first, second))
    logger.info("Determinism (%s retrained with the same seed): %s", key, deterministic)
    result = {
        "model": args.model,
        "condition": condition,
        "seed": args.seed,
        "config": config,
        "git_commit": git_commit(),
        "git_dirty": dirty,
        "deterministic": deterministic,
        "determinism_group": "/".join(key),
        "groups": groups,
        "environment": environment_info(),
    }
    (out / "controls.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if deterministic else 1


if __name__ == "__main__":
    raise SystemExit(main())
