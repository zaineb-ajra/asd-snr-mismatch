"""Fidelity check of the Wilkinghoff re-implementation (not part of the study).

Criteria pre-registered in docs/validation.md ("Wilkinghoff"). This module holds
the DCASE data loaders, the metrics under both per-domain definitions, and a
``timing`` command that trains for one epoch on the DCASE 2022 development
training data and reports the time, without computing any metric.

Usage::

    CUDA_VISIBLE_DEVICES=1 python -m noisemismatch.verification_wilkinghoff timing
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import re
import time
import zipfile
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats
import soundfile as sf
import torch

from noisemismatch.config import CONFIG_DIR, load_paths, load_yaml
from noisemismatch.data.dataset import ClipSet, select
from noisemismatch.data.prepare import METADATA_DIR
from noisemismatch.data.split import ANOMALOUS, FILE_COLUMNS, NORMAL, TEST, TRAIN
from noisemismatch.eval.metrics import auc, pauc
from noisemismatch.models.wilkinghoff import WilkinghoffDetector, adjust_length
from noisemismatch.utils.logs import setup_logging
from noisemismatch.utils.provenance import environment_info, git_commit, git_is_dirty
from noisemismatch.utils.repro import seed_everything
from noisemismatch.verification_dcase2022 import DcaseAudio, clip_table

logger = logging.getLogger(__name__)

DCASE2022_TYPES: tuple[str, ...] = (
    "ToyCar",
    "ToyTrain",
    "bearing",
    "fan",
    "gearbox",
    "slider",
    "valve",
)
ATTRIBUTES: re.Pattern[str] = re.compile(r"_\d{4}_(?P<attributes>.+)$")


def with_classes(table: pd.DataFrame) -> pd.DataFrame:
    """Adds ``class_label`` = machine type / section / attribute values."""
    attributes = table["clip"].str.extract(ATTRIBUTES)["attributes"].fillna("")
    out = table.copy()
    out["class_label"] = (
        out["machine_type"] + "/" + out["machine_id"] + "/" + attributes
    )
    return out


class MultiAudio:
    """Audio source over several single-type DCASE archives."""

    def __init__(self, sources: dict[str, DcaseAudio]) -> None:
        self.sources = sources
        self.fingerprint = "|".join(s.fingerprint for s in sources.values())

    def index(self, machine_type: str, snr: int) -> pd.DataFrame:
        """File columns of one machine type."""
        return self.sources[machine_type].index(machine_type, snr)

    def array(self, machine_type: str, snr: int) -> np.ndarray:
        """Waveforms of one machine type."""
        return self.sources[machine_type].array(machine_type, snr)

    def load(self, machine_type: str, snr: int, files: pd.DataFrame) -> np.ndarray:
        """Waveforms of the given files of one machine type."""
        return self.sources[machine_type].load(machine_type, snr, files)


def dcase2022_dev(root: Path) -> tuple[pd.DataFrame, MultiAudio]:
    """Clip table (with classes) and audio of the DCASE 2022 development set."""
    tables, sources = [], {}
    for machine_type in DCASE2022_TYPES:
        archive = root / f"dev_{machine_type}.zip"
        table = clip_table(archive)
        sources[machine_type] = DcaseAudio(archive, table)
        tables.append(table)
    return with_classes(pd.concat(tables, ignore_index=True)), MultiAudio(sources)


NAME: re.Pattern[str] = re.compile(
    r"(?P<machine_type>[^/]+)/(?P<split>train|test)/(?P<clip>section_(?P<section>\d{2})"
    r"(?:_(?P<domain>source|target)_(?:train|test)_(?P<label>normal|anomaly))?"
    r"_\d{4}(?:_(?P<attributes>.+))?)\.wav"
)
DATASETS: dict[str, dict[str, Any]] = {
    "2022": {
        "dir": "dcase2022",
        "input_length": 160000,
        "archives": [
            "dev_{t}.zip",
            "eval_data_{t}_train.zip",
            "eval_data_{t}_test.zip",
        ],
        "types": DCASE2022_TYPES,
        "eval_sections": ("03", "04", "05"),
    },
    "2023": {
        "dir": "dcase2023",
        "input_length": 288000,
        "archives": ["dev_{t}.zip", "eval_data_{t}_train.zip"],
        "types": (
            "ToyCar",
            "ToyTrain",
            "bearing",
            "fan",
            "gearbox",
            "slider",
            "valve",
            "ToyDrone",
            "ToyNscale",
            "ToyTank",
            "Vacuum",
            "bandsaw",
            "grinder",
            "shaker",
        ),
        "eval_sections": (),
    },
}


def archive_table(archive: Path) -> pd.DataFrame:
    """Rows of the WAV members of one DCASE archive (labels may be unknown)."""
    rows = []
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            match = NAME.fullmatch(member)
            if match is not None:
                rows.append(
                    {**match.groupdict(), "member": member, "archive": str(archive)}
                )
    return pd.DataFrame(rows)


def ground_truth(root: Path, machine_type: str, section: str) -> pd.DataFrame:
    """Labels and domains of the DCASE 2022 evaluation test clips of one section."""
    base = root / "ground_truth"
    (commit,) = [d for d in base.iterdir() if d.is_dir()]
    name = f"ground_truth_{machine_type}_section_{section}_test.csv"
    labels = pd.read_csv(
        commit / "ground_truth_data" / name, header=None, names=["file", "y"]
    )
    domains = pd.read_csv(
        commit / "ground_truth_domain" / name, header=None, names=["file", "d"]
    )
    out = labels.merge(domains, on="file", validate="one_to_one")
    out["clip"] = out["file"].str.removesuffix(".wav")
    out["domain"] = out["d"].map({0: "source", 1: "target"})
    return out[["clip", "y", "domain"]]


def dataset_table(dataset: str, data_root: Path) -> pd.DataFrame:
    """Clip table of a fidelity dataset.

    Training clips of the development and evaluation sets, and the scored test
    clips (column ``set``: dev or eval).

    Returns:
        Rows with the clip-table columns, ``domain``, ``set``, ``member``,
        ``archive`` and, for training clips, ``class_label``.
    """
    spec = DATASETS[dataset]
    root = data_root / spec["dir"]
    parts = []
    for machine_type in spec["types"]:
        for pattern in spec["archives"]:
            path = root / pattern.format(t=machine_type)
            if path.exists():
                parts.append(archive_table(path))
    table = pd.concat(parts, ignore_index=True)
    table["machine_id"] = "section_" + table["section"]
    table["set"] = np.where(table["section"].isin(spec["eval_sections"]), "eval", "dev")
    test_eval = (table["split"] == "test") & (table["set"] == "eval")
    if test_eval.any():
        truth = pd.concat(
            [
                ground_truth(root, t, s).assign(machine_type=t, section=s)
                for t, s in table.loc[test_eval, ["machine_type", "section"]]
                .drop_duplicates()
                .itertuples(index=False)
            ],
            ignore_index=True,
        )
        merged = (
            table[test_eval]
            .drop(columns=["domain"])
            .merge(
                truth,
                on=["machine_type", "section", "clip"],
                how="left",
                validate="one_to_one",
            )
        )
        if merged["y"].isna().any():
            raise ValueError("evaluation test clips without ground truth")
        table.loc[test_eval, "domain"] = merged["domain"].to_numpy()
        table.loc[test_eval, "label"] = np.where(
            merged["y"].to_numpy() == 1, "anomaly", "normal"
        )
    # DCASE 2023 evaluation machine types: training data only (ground truth excluded)
    table = table[(table["split"] == "train") | (table["label"].notna())].reset_index(
        drop=True
    )
    table["label"] = table["label"].map({"normal": NORMAL, "anomaly": ANOMALOUS})
    table["y"] = (table["label"] == ANOMALOUS).astype(int)
    table["snr_db"] = 0
    table["file_machine_id"] = table["machine_id"]
    table["file_label"] = table["label"]
    table["file_clip"] = table["clip"]
    table["strict_exclude"] = False
    table["split"] = np.where(table["split"] == "train", TRAIN, TEST)
    attributes = table["attributes"].fillna("")
    table["class_label"] = (
        table["machine_type"] + "/" + table["machine_id"] + "/" + attributes
    )
    return table


class ArchiveAudio:
    """Waveforms of the DCASE clips of a table, per machine type, at a fixed length.

    Clips shorter or longer than ``length`` are repeated and randomly cropped once,
    with a generator seeded by (seed, CRC-32 of the member name).
    """

    def __init__(
        self, table: pd.DataFrame, length: int, seed: int, extended: bool = False
    ) -> None:
        self.table = table
        self.length, self.seed = length, seed
        self.extended = extended
        self.stored = 2 * length if extended else length
        self.fingerprint = f"dcase-fidelity:{length}:{seed}:{extended}"
        self._arrays: dict[str, np.ndarray] = {}
        self._native: dict[str, np.ndarray] = {}
        self._rows: dict[str, dict[tuple, int]] = {}
        if extended:  # the model then crops at every training step
            self.native_lengths = self._native_lengths

    def _load_type(self, machine_type: str) -> None:
        if machine_type in self._arrays:
            return
        part = self.table[self.table["machine_type"] == machine_type].reset_index(
            drop=True
        )
        out = np.empty((len(part), self.stored), dtype=np.int16)
        native = np.empty(len(part), dtype=np.int64)
        for archive, rows in part.groupby("archive"):
            with zipfile.ZipFile(archive) as zf:
                for k, member in zip(rows.index, rows["member"], strict=True):
                    with zf.open(member) as f:
                        wave, rate = sf.read(f, dtype="int16")
                    if rate != 16000 or wave.ndim != 1:
                        raise ValueError(f"{member}: unexpected format")
                    native[k] = len(wave)
                    if self.extended:  # periodic extension, cropped later
                        reps = math.ceil(self.stored / len(wave))
                        out[k] = np.tile(wave, reps)[: self.stored]
                    else:
                        rng = np.random.default_rng(
                            [self.seed, zlib.crc32(member.encode())]
                        )
                        out[k] = adjust_length(wave, self.length, rng)
        self._arrays[machine_type] = out
        self._native[machine_type] = native
        self._rows[machine_type] = {
            tuple(r): n for n, r in enumerate(part[list(FILE_COLUMNS)].to_numpy())
        }

    def load(self, machine_type: str, snr: int, files: pd.DataFrame) -> np.ndarray:
        """Waveforms of the given files of one machine type, in order."""
        self._load_type(machine_type)
        rows = self._rows[machine_type]
        order = [rows[tuple(r)] for r in files[list(FILE_COLUMNS)].to_numpy()]
        return self._arrays[machine_type][order]

    def _native_lengths(
        self, machine_type: str, snr: int, files: pd.DataFrame
    ) -> np.ndarray:
        """Native lengths (samples) of the given files of one machine type."""
        self._load_type(machine_type)
        rows = self._rows[machine_type]
        order = [rows[tuple(r)] for r in files[list(FILE_COLUMNS)].to_numpy()]
        return self._native[machine_type][order]


def section_metrics(scores: pd.DataFrame, max_fpr: float) -> pd.DataFrame:
    """Per-section metrics under both per-domain definitions (x 100).

    Args:
        scores: Test rows with ``machine_type``, ``machine_id`` (section),
            ``domain``, ``y`` and ``score``.
        max_fpr: pAUC bound.

    Returns:
        One row per section: ``auc_mixed``, ``pauc_mixed`` (all clips), and for
        each domain ``d`` and definition ``official`` (normal clips of ``d`` and
        all anomalous clips) or ``restricted`` (clips of ``d`` only):
        ``auc_{d}_{definition}``, ``pauc_{d}_{definition}``.
    """
    rows = []
    for (machine_type, section), part in scores.groupby(["machine_type", "machine_id"]):
        y, s = part["y"].to_numpy(), part["score"].to_numpy()
        row = {
            "machine_type": machine_type,
            "section": section,
            "auc_mixed": 100 * auc(y, s),
            "pauc_mixed": 100 * pauc(y, s, max_fpr),
        }
        for domain in ("source", "target"):
            in_domain = (part["domain"] == domain).to_numpy()
            for name, keep in (
                ("official", in_domain | (y == 1)),
                ("restricted", in_domain),
            ):
                row[f"auc_{domain}_{name}"] = 100 * auc(y[keep], s[keep])
                row[f"pauc_{domain}_{name}"] = 100 * pauc(y[keep], s[keep], max_fpr)
        rows.append(row)
    return pd.DataFrame(rows)


def harmonic_means(sections: pd.DataFrame) -> dict[str, float]:
    """Harmonic mean over the sections of every metric column."""
    columns = [c for c in sections.columns if c.startswith(("auc_", "pauc_"))]
    return {c: float(scipy.stats.hmean(sections[c].clip(lower=1e-12))) for c in columns}


def run_timing(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    """One epoch of training on the DCASE 2022 development training data.

    No test clip is scored and no metric is computed.
    """
    paths = load_paths()
    table, audio = dcase2022_dev(paths.data_root / "dcase2022")
    train = ClipSet(select(table, TRAIN), audio, paths.cache_root)
    config = copy.deepcopy(load_yaml(CONFIG_DIR / "models" / "wilkinghoff.yaml"))
    config["epochs"] = 1
    seed_everything(0)
    started = time.time()
    train.raw_waveforms()  # load the archives once (not part of the training time)
    loaded = time.time()
    model = WilkinghoffDetector(config, 0, torch.device(args.device))
    model.fit(train)
    fitted = time.time()
    n_classes = len(model.classes)
    return {
        "n_train_clips": len(train),
        "n_classes_dev": n_classes,
        "steps": model.steps,
        "load_s": loaded - started,
        "fit_s": fitted - loaded,
        "seconds_per_step_incl_backend": (fitted - loaded) / model.steps,
        "history": model.history,
    }


PUBLISHED: Path = METADATA_DIR / "published" / "wilkinghoff2024_table1.csv"
TOLERANCE: float = 3.0
# Pre-registered cells: (domain, metric) whose metric has a published definition.
CRITERION: tuple[tuple[str, str, str], ...] = (
    ("source", "auc", "auc_source_official"),
    ("target", "auc", "auc_target_official"),
    ("mixed", "auc", "auc_mixed"),
    ("mixed", "pauc", "pauc_mixed"),
)
INFORMATION: tuple[tuple[str, str, str], ...] = (
    ("source", "auc", "auc_source_restricted"),
    ("target", "auc", "auc_target_restricted"),
    ("source", "pauc", "pauc_source_official"),
    ("source", "pauc", "pauc_source_restricted"),
    ("target", "pauc", "pauc_target_official"),
    ("target", "pauc", "pauc_target_restricted"),
)


def compare(summary: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Published cells against our means, for both kinds of cells.

    Args:
        summary: Mean and std over seeds per (set, metric column).
        dataset: ``2022`` or ``2023``.

    Returns:
        One row per (published cell, our metric column), with ``criterion`` True
        for the pre-registered cells and ``agrees`` for those.
    """
    published = pd.read_csv(PUBLISHED)
    rows = []
    for (domain, metric, column), criterion in [(c, True) for c in CRITERION] + [
        (c, False) for c in INFORMATION
    ]:
        for part in ("dev", "eval"):
            name = f"dcase{dataset}_{part}"
            cell = published[
                (published["set"] == name)
                & (published["domain"] == domain)
                & (published["metric"] == metric)
            ]
            ours = summary[(summary["set"] == part) & (summary["metric"] == column)]
            if cell.empty or ours.empty:
                continue
            mean = float(ours["mean"].iloc[0])
            difference = mean - float(cell["mean"].iloc[0])
            rows.append(
                {
                    "set": name,
                    "published_cell": f"{metric} {domain}",
                    "our_metric": column,
                    "criterion": criterion,
                    "published_mean": float(cell["mean"].iloc[0]),
                    "published_std": float(cell["std"].iloc[0]),
                    "our_mean": mean,
                    "our_std": float(ours["std"].iloc[0]),
                    "n_seeds": int(ours["n"].iloc[0]),
                    "difference": difference,
                    "agrees": bool(abs(difference) <= TOLERANCE) if criterion else None,
                }
            )
    return pd.DataFrame(rows)


def test_only_classes(table: pd.DataFrame) -> list[str]:
    """Classes named by development test clips but absent from the training data.

    The author's class counts (342 for DCASE 2022, 186 for DCASE 2023) are
    reproduced exactly when the attribute values in the file names of the
    development test clips are included (19 extra classes in each dataset).
    """
    named = table[(table["split"] == TEST) & table["attributes"].notna()]
    train = set(table.loc[table["split"] == TRAIN, "class_label"])
    return sorted(set(named["class_label"]) - train)


def run_classes(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    """Number of training classes of a dataset (test of hypothesis H)."""
    table = dataset_table(args.dataset, load_paths().data_root)
    train = table[table["split"] == TRAIN]
    return {
        "dataset": args.dataset,
        "n_train_clips": len(train),
        "n_classes": int(train["class_label"].nunique()),
        "n_test_only_classes": len(test_only_classes(table)),
        "n_classes_with_test_only": int(train["class_label"].nunique())
        + len(test_only_classes(table)),
        "n_classes_per_set": train.groupby("set")["class_label"].nunique().to_dict(),
        "n_test_clips_per_set": table[table["split"] == TEST]
        .groupby("set")
        .size()
        .to_dict(),
    }


def run_fidelity(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    """Five seeds of one dataset; per-clip scores, both metric definitions."""
    paths = load_paths()
    spec = DATASETS[args.dataset]
    protocol = load_yaml(CONFIG_DIR / "protocol.yaml")
    max_fpr = float(protocol["evaluation"]["pauc_max_fpr"])
    config = copy.deepcopy(load_yaml(CONFIG_DIR / "models" / "wilkinghoff.yaml"))
    config["input_length"] = spec["input_length"]
    table = dataset_table(args.dataset, paths.data_root)
    audio = ArchiveAudio(
        table, spec["input_length"], seed=0, extended=args.crop == "per-step"
    )
    train = ClipSet(select(table, TRAIN), audio)
    test = ClipSet(select(table, TEST), audio)
    commit = git_commit()
    extra = test_only_classes(table)
    rows = []
    for seed in args.seeds:
        run_dir = out / f"seed{seed}"
        done = run_dir / "result.json"
        if not done.exists():
            run_dir.mkdir(parents=True, exist_ok=True)
            started = time.time()
            seed_everything(seed)
            model = WilkinghoffDetector(config, seed, torch.device(args.device))
            model.extra_classes = extra
            model.fit(train)
            scores = test.table[
                ["set", "machine_type", "machine_id", "domain", "label", "clip", "y"]
            ].copy()
            scores["score"] = model.score(test)
            scores.to_csv(run_dir / "scores.csv", index=False, lineterminator="\n")
            result: dict[str, Any] = {
                "dataset": args.dataset,
                "seed": seed,
                "config": config,
                "git_commit": commit,
                "n_classes": len(model.classes),
                "steps": model.steps,
                "history": model.history,
                "sets": {},
            }
            for part, sub in scores.groupby("set"):
                sections = section_metrics(sub, max_fpr)
                result["sets"][part] = {
                    "harmonic_means": harmonic_means(sections),
                    "sections": sections.to_dict("records"),
                }
            result["duration_s"] = time.time() - started
            done.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            logger.info("%s seed %d: %.0f s", args.dataset, seed, result["duration_s"])
        result = json.loads(done.read_text())
        for part, values in result["sets"].items():
            for metric, value in values["harmonic_means"].items():
                rows.append(
                    {"set": part, "metric": metric, "seed": seed, "value": value}
                )
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(out / "per_seed.csv", index=False, lineterminator="\n")
    summary = (
        per_seed.groupby(["set", "metric"])["value"]
        .agg(mean="mean", std=lambda v: v.std(ddof=1), n="size")
        .reset_index()
    )
    summary.to_csv(
        out / "summary.csv", index=False, lineterminator="\n", float_format="%.3f"
    )
    comparison = compare(summary, args.dataset)
    comparison.to_csv(
        out / "comparison.csv", index=False, lineterminator="\n", float_format="%.3f"
    )
    logger.info("Comparison:\n%s", comparison.round(2).to_string(index=False))
    return {"n_seeds": len(args.seeds)}


def main(argv: list[str] | None = None) -> int:
    """Runs one command.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Wilkinghoff fidelity check.")
    parser.add_argument("command", choices=["timing", "classes", "fidelity"])
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="2022")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--crop",
        choices=["once", "per-step"],
        default="once",
        help="Clips of unequal length: cropped once at loading (default) or at "
        "random at every training step (control, written apart).",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    paths = load_paths()
    out = paths.output_root / "verification" / "wilkinghoff" / args.command
    if args.command != "timing":
        suffix = "" if args.crop == "once" else "_crop_per_step"
        out = out / f"dcase{args.dataset}{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / f"{args.command}.log")
    dirty = git_is_dirty()
    if dirty and not args.allow_dirty:
        logger.error("Uncommitted changes: commit them or pass --allow-dirty")
        return 1
    run = {"timing": run_timing, "classes": run_classes, "fidelity": run_fidelity}
    result = run[args.command](args, out)
    result.update(
        {
            "git_commit": git_commit(),
            "git_dirty": dirty,
            "environment": environment_info(),
        }
    )
    (out / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(
        "%s: %s", args.command, {k: v for k, v in result.items() if k != "environment"}
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
