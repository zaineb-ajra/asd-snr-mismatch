"""Phase 7, step 1: per-ID metrics of the study runs and EAT scoring variants.

Usage::

    python scripts/study_analysis.py per-id --workers 16
    python scripts/study_analysis.py eat-variants
    python scripts/study_analysis.py bootstrap --workers 15

``per-id`` reads the per-clip scores of the Phase 6 runs and writes, under
``results/aggregated/``: ``per_id_main.csv`` and ``per_id_sensitivity.csv``
(AUC, pAUC and the pre-registered 95 % bootstrap interval of the AUC per run,
test SNR and machine ID) and ``per_id_main_strict.csv`` (strict identity
variant, no interval). It checks that every AUC and pAUC equals the value
recorded by the run in ``result.json``.

``eat-variants`` computes the pre-registered sensitivity analysis of the EAT
scoring (pooling x K, docs/protocol.md) for every training condition and test
SNR of the main split, from the cached embeddings, into
``results/aggregated/eat_variants_main.csv``; it checks that the main variant
reproduces the AUCs of the study runs.

``bootstrap`` computes the interval on aggregated values decided after the
results (docs/protocol.md, "Analyses added after the results"): paired bootstrap
of the test clips, the same draws for every model, condition, seed and test SNR,
into ``results/aggregated/bootstrap_intervals_<split>.csv`` (ΔAUC averaged over
seeds, ratio of errors, direction of the larger loss, paired differences between
models).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from noisemismatch.config import CONFIG_DIR, REPO_ROOT, load_paths, load_yaml
from noisemismatch.data.audio_cache import AudioCache, cache_dir
from noisemismatch.data.build_split import identity_sets
from noisemismatch.data.dataset import ClipSet, build_clip_table, select
from noisemismatch.data.identity import load_identity
from noisemismatch.data.split import TEST, TRAIN, load_split
from noisemismatch.eval.metrics import auc, pauc
from noisemismatch.eval.study import (
    COLUMN_KEYS,
    CONDITION_SNR,
    MODELS,
    delta_over_seeds,
    interval,
    load_scores,
    mean_over_seeds,
    per_id,
    ratio_over_seeds,
    resample_ids,
)
from noisemismatch.models.eat_knn import VARIANTS, EatKnnDetector
from noisemismatch.run import SPLIT_FILES
from noisemismatch.utils.logs import setup_logging

logger = logging.getLogger("study_analysis")

OUT = REPO_ROOT / "results" / "aggregated"
KEYS: list[str] = [
    "model",
    "train_condition",
    "seed",
    "test_snr_db",
    "machine_type",
    "machine_id",
]


def check_against_runs(table: pd.DataFrame, runs_dir: Path, split: str) -> float:
    """Largest difference between the recomputed and the recorded metrics."""
    recorded = []
    for path in sorted((runs_dir / split).glob("*/*/seed*/result.json")):
        result = json.loads(path.read_text())
        frame = pd.DataFrame(result["metrics"])
        frame["model"] = result["model"]
        frame["train_condition"] = result["train_condition"]
        frame["seed"] = result["seed"]
        recorded.append(frame)
    merged = table.merge(
        pd.concat(recorded), on=KEYS, suffixes=("", "_run"), validate="one_to_one"
    )
    if len(merged) != len(table):
        raise ValueError(f"{split}: {len(table)} rows, {len(merged)} matched")
    return float(
        max(
            (merged["auc"] - merged["auc_run"]).abs().max(),
            (merged["pauc"] - merged["pauc_run"]).abs().max(),
        )
    )


def per_id_command(workers: int) -> None:
    """Per-ID metrics with intervals (main, sensitivity) and strict variant."""
    paths = load_paths()
    runs = paths.output_root / "runs"
    protocol = load_yaml(CONFIG_DIR / "protocol.yaml")
    evaluation = protocol["evaluation"]
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("main", "sensitivity"):
        scores = load_scores(runs, split)
        table = per_id(
            scores, evaluation["pauc_max_fpr"], evaluation["bootstrap"], workers=workers
        )
        gap = check_against_runs(table, runs, split)
        logger.info(
            "%s: %d per-ID rows; largest gap to the runs %.2e", split, len(table), gap
        )
        if gap > 1e-12:
            raise ValueError(
                f"{split}: recomputed metrics differ from the runs ({gap})"
            )
        table.to_csv(OUT / f"per_id_{split}.csv", index=False, float_format="%.6f")
        if split == "main":
            strict = per_id(scores, evaluation["pauc_max_fpr"], None, strict=True)
            strict.to_csv(
                OUT / "per_id_main_strict.csv", index=False, float_format="%.6f"
            )
            logger.info("main strict: %d per-ID rows", len(strict))


def eat_variants_command() -> None:
    """EAT scoring variants for every condition of the main split."""
    paths = load_paths()
    protocol = load_yaml(CONFIG_DIR / "protocol.yaml")
    config = load_yaml(CONFIG_DIR / "models" / "eat_knn.yaml")
    max_fpr = float(protocol["evaluation"]["pauc_max_fpr"])
    snrs = tuple(protocol["evaluation"]["test_snrs_db"])
    sets = identity_sets(load_identity(), protocol["dataset"]["excluded_machine_ids"])
    table = build_clip_table(
        load_split(SPLIT_FILES["main"]),
        sets["main"],
        snrs,
        protocol["evaluation"]["strict_variant"],
    )
    audio = AudioCache(cache_dir(paths.cache_root))
    device = torch.device("cpu")
    test = ClipSet(select(table, TEST, snrs), audio, paths.cache_root)
    test_groups = dict(test.groups("machine_id"))
    rows = []
    for condition, snr in CONDITION_SNR.items():
        train_snrs = snrs if snr is None else (snr,)
        train = ClipSet(select(table, TRAIN, train_snrs), audio, paths.cache_root)
        for key, train_group in train.groups("machine_id"):
            model = EatKnnDetector(config, 0, device)
            model.fit(train_group)
            for test_snr in snrs:
                group = test_groups[key].subset(
                    test_groups[key].table["snr_db"] == test_snr
                )
                y = group.table["y"].to_numpy()
                for (pooling, k), scores in model.score_variants(
                    group, VARIANTS
                ).items():
                    rows.append(
                        {
                            "train_condition": condition,
                            "test_snr_db": test_snr,
                            "machine_type": key[0],
                            "machine_id": key[1],
                            "pooling": pooling,
                            "k": k,
                            "main": (pooling, k)
                            == (config["pooling"], int(config["k"])),
                            "auc": auc(y, scores),
                            "pauc": pauc(y, scores, max_fpr),
                        }
                    )
        logger.info("EAT variants: %s done", condition)
    variants = pd.DataFrame(rows)
    recorded = []
    for condition in CONDITION_SNR:
        result = json.loads(
            (
                paths.output_root
                / "runs/main/eat_knn"
                / condition
                / "seed0/result.json"
            ).read_text()
        )
        recorded.append(
            pd.DataFrame(result["metrics"]).assign(train_condition=condition)
        )
    study = pd.concat(recorded)  # full precision, not the rounded CSV
    merged = variants[variants["main"]].merge(
        study,
        on=["train_condition", "test_snr_db", "machine_type", "machine_id"],
        suffixes=("", "_run"),
        validate="one_to_one",
    )
    gap = float(np.abs(merged["auc"] - merged["auc_run"]).max())
    logger.info(
        "EAT main variant: %d cells, largest gap to the runs %.2e", len(merged), gap
    )
    if len(merged) != 4 * 45 or gap > 1e-9:
        raise ValueError(f"main variant does not reproduce the study runs ({gap})")
    variants.to_csv(OUT / "eat_variants_main.csv", index=False, float_format="%.6f")


def _row(
    statistic: str,
    model: str,
    other: str,
    cell: tuple,
    point: float,
    samples: np.ndarray,
    confidence: float,
) -> dict:
    """One interval of ``bootstrap_intervals_<split>.csv``."""
    return {
        "statistic": statistic,
        "model": model,
        "model_b": other,
        "train_condition": cell[0],
        "test_snr_db": cell[1],
    } | interval(point, samples, confidence)


def _intervals(
    point: pd.DataFrame, boot: pd.DataFrame, metric: str, full: bool, confidence: float
) -> list[dict]:
    """Intervals of one evaluation and metric (see ``bootstrap``)."""
    snrs = sorted({c[3] for c in point.columns})
    conditions = sorted({c[1] for c in point.columns}, key=list(CONDITION_SNR).index)
    rows = []
    for condition in conditions:
        for snr in snrs:
            for model in MODELS:
                rows.append(
                    _row(
                        "level",
                        model,
                        "",
                        (condition, snr),
                        mean_over_seeds(point, model, condition, snr)[0],
                        mean_over_seeds(boot, model, condition, snr),
                        confidence,
                    )
                )
    if full:  # order of the models in the matched cells
        for condition in conditions:
            snr = CONDITION_SNR[condition]
            if snr is None:
                continue
            values = {
                m: (
                    mean_over_seeds(point, m, condition, snr)[0],
                    mean_over_seeds(boot, m, condition, snr),
                )
                for m in MODELS
            }
            for i, model in enumerate(MODELS):
                for other in MODELS[i + 1 :]:
                    rows.append(
                        _row(
                            "level_difference",
                            model,
                            other,
                            (condition, snr),
                            values[model][0] - values[other][0],
                            values[model][1] - values[other][1],
                            confidence,
                        )
                    )
    cells = [(c, snr) for c in conditions for snr in snrs if CONDITION_SNR[c] != snr]
    measures = {"delta": delta_over_seeds}
    if full and metric == "auc":
        measures["ratio"] = ratio_over_seeds
    for name, function in measures.items():
        for cell in cells:
            values = {
                m: (function(point, m, *cell)[0], function(boot, m, *cell))
                for m in MODELS
            }
            for model in MODELS:
                rows.append(_row(name, model, "", cell, *values[model], confidence))
            if not full:
                continue
            for i, model in enumerate(MODELS):
                for other in MODELS[i + 1 :]:
                    rows.append(
                        _row(
                            f"{name}_difference",
                            model,
                            other,
                            cell,
                            values[model][0] - values[other][0],
                            values[model][1] - values[other][1],
                            confidence,
                        )
                    )
    if full and metric == "auc" and "m6dB" in conditions:
        # ΔAUC against the SNR gap (test - training) for the figure: cells with
        # the same gap averaged seed by seed (protocol, analyses after results).
        for model in MODELS:
            for gap in (-12, -6, 6, 12):
                same = [
                    (c, snr)
                    for c, snr in cells
                    if CONDITION_SNR[c] is not None and snr - CONDITION_SNR[c] == gap
                ]
                rows.append(
                    _row(
                        "gap",
                        model,
                        "",
                        ("gap", gap),
                        float(
                            np.mean(
                                [delta_over_seeds(point, model, *c)[0] for c in same]
                            )
                        ),
                        np.mean(
                            [delta_over_seeds(boot, model, *c) for c in same], axis=0
                        ),
                        confidence,
                    )
                )
    if full and "m6dB" in conditions:
        for model in MODELS:
            clean = (
                delta_over_seeds(point, model, "m6dB", 6)[0],
                delta_over_seeds(boot, model, "m6dB", 6),
            )
            noisy = (
                delta_over_seeds(point, model, "p6dB", -6)[0],
                delta_over_seeds(boot, model, "p6dB", -6),
            )
            rows.append(
                _row(
                    "direction",
                    model,
                    "",
                    ("m6dB-p6dB", "6/-6"),
                    clean[0] - noisy[0],
                    clean[1] - noisy[1],
                    confidence,
                )
            )
    return rows


def bootstrap_command(workers: int) -> None:
    """Paired bootstrap intervals on aggregated values.

    Main split: main evaluation, strict identity variant and aggregation without
    fan id_00; sensitivity split: main evaluation and aggregation without fan
    id_00. For each: the aggregated AUC and pAUC of every cell (mean over seeds)
    and the seed-paired ΔAUC and ΔpAUC; for the main evaluation of each split, the
    ratio of errors, the paired differences between models and (main split) the
    direction of the larger loss.
    """
    paths = load_paths()
    runs = paths.output_root / "runs"
    protocol = load_yaml(CONFIG_DIR / "protocol.yaml")
    settings = protocol["evaluation"]["bootstrap"]
    n, confidence, seed = (
        int(settings["n_resamples"]),
        float(settings["confidence"]),
        int(settings["seed"]),
    )
    max_fpr = float(protocol["evaluation"]["pauc_max_fpr"])
    drop = protocol["aggregation"]["robustness"][0]["drop"]
    recorded = pd.read_csv(OUT / "aggregated.csv")
    for split in ("main", "sensitivity"):
        scores = load_scores(runs, split)
        evaluations = {"main": (scores, ())}
        evaluations["fan_without_id_00"] = (scores, drop)
        if split == "main":
            evaluations["strict"] = (scores[~scores["strict_exclude"].astype(bool)], ())
        cache: dict[int, object] = {}
        rows = []
        for evaluation, (table, dropped) in evaluations.items():
            key = id(table)
            if key not in cache:
                cache[key] = resample_ids(table, n, seed, workers, max_fpr)
            resamples = cache[key]
            name = (
                split
                if evaluation == "main"
                else (evaluation if split == "main" else f"{split}_{evaluation}")
            )
            for metric in ("auc", "pauc"):
                point, boot = resamples.aggregate(metric, dropped)
                check = recorded[recorded["aggregation"] == name].set_index(
                    COLUMN_KEYS
                )[metric]
                gap = max(abs(point[c].item() - check[c]) for c in point.columns)
                logger.info(
                    "%s %s %s: point estimates against aggregated.csv, largest gap %.1e",
                    split,
                    evaluation,
                    metric,
                    gap,
                )
                if gap > 1e-6:  # aggregated.csv has six decimals
                    raise ValueError(f"{name} {metric}: point estimates differ ({gap})")
                for row in _intervals(
                    point, boot, metric, evaluation == "main", confidence
                ):
                    rows.append({"evaluation": evaluation, "metric": metric} | row)
            if evaluation == "main":
                for machine_type in sorted({t for t, _ in resamples.groups}):
                    point, boot = resamples.machine_type("auc", machine_type)
                    for row in _intervals(point, boot, "auc", False, confidence):
                        rows.append(
                            {"evaluation": f"type:{machine_type}", "metric": "auc"}
                            | row
                        )
        table = pd.DataFrame(rows)
        table.to_csv(
            OUT / f"bootstrap_intervals_{split}.csv", index=False, float_format="%.6f"
        )
        logger.info("%s: %d intervals written", split, len(table))


def main(argv: list[str] | None = None) -> int:
    """Runs one step of the analysis.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        0 on success.
    """
    parser = argparse.ArgumentParser(description="Phase 7 analysis, step 1.")
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("per-id")
    one.add_argument("--workers", type=int, default=1)
    sub.add_parser("eat-variants")
    boot = sub.add_parser("bootstrap")
    boot.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    setup_logging()
    torch.set_num_threads(8)
    if args.command == "per-id":
        per_id_command(args.workers)
    elif args.command == "bootstrap":
        bootstrap_command(args.workers)
    else:
        eat_variants_command()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
