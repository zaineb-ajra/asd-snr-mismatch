"""Fidelity check of the MobileNetV2 port on DCASE 2022 Task 2 (not part of the study).

Runs the ported baseline with its original configuration (including
``validation_split=0.1``) on the DCASE 2022 Task 2 development set, 5 seeds per
machine type, computes the metrics of the baseline code (``01_test.py``) and
compares them with the published cells selected in docs/validation.md (published
standard deviation <= 3.0 points; agreement if |our mean - published mean| <= 3.0).

Usage::

    CUDA_VISIBLE_DEVICES=1 python -m noisemismatch.verification_dcase2022 [--types fan valve]

The per-section metrics follow ``01_test.py`` of
Kota-Dohi/dcase2022_task2_baseline_mobile_net_v2 (commit aaf3932; MIT License,
Copyright (c) 2022 Kota-Dohi; terms in THIRD_PARTY_NOTICES.md).

Data: ``<data_root>/dcase2022/dev_<type>.zip`` (Zenodo 6355122, CC BY 4.0).
Outputs: ``<output_root>/verification/mobilenetv2/dcase2022/``.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from noisemismatch.config import CONFIG_DIR, REPO_ROOT, load_paths, load_yaml
from noisemismatch.data.dataset import ClipSet, select
from noisemismatch.data.split import ANOMALOUS, FILE_COLUMNS, NORMAL, TEST, TRAIN
from noisemismatch.eval.metrics import auc, pauc
from noisemismatch.models.mobilenetv2 import MobileNetV2Detector
from noisemismatch.utils.logs import setup_logging
from noisemismatch.utils.provenance import environment_info, git_commit, git_is_dirty
from noisemismatch.utils.repro import seed_everything

logger = logging.getLogger(__name__)

PUBLISHED: Path = (
    REPO_ROOT / "metadata" / "published" / "dohi2022_table2_mobilenetv2.csv"
)
MAX_STD: float = 3.0
TOLERANCE: float = 3.0
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
MEMBER: re.Pattern[str] = re.compile(
    r"(?P<machine_type>\w+)/(?P<split>train|test)/(?P<clip>section_(?P<section>\d{2})_"
    r"(?P<domain>source|target)_(?:train|test)_(?P<label>normal|anomaly)_\d{4}_.+)\.wav"
)


def clip_table(archive: Path) -> pd.DataFrame:
    """Clip table of one DCASE 2022 archive (sections play the role of IDs).

    Args:
        archive: ``dev_<type>.zip``.

    Returns:
        Clip-table rows (``machine_id`` = ``section_XX``, ``snr_db`` = 0) with a
        ``domain`` column, sorted like the baseline's file lists.
    """
    rows = []
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            match = MEMBER.fullmatch(name)
            if match is None:
                continue  # attribute CSV files
            label = NORMAL if match["label"] == "normal" else ANOMALOUS
            rows.append(
                {
                    "machine_type": match["machine_type"],
                    "machine_id": f"section_{match['section']}",
                    "label": label,
                    "clip": match["clip"],
                    "snr_db": 0,
                    "file_machine_id": f"section_{match['section']}",
                    "file_label": label,
                    "file_clip": match["clip"],
                    "split": TRAIN if match["split"] == "train" else TEST,
                    "y": int(label == ANOMALOUS),
                    "strict_exclude": False,
                    "domain": match["domain"],
                    "member": name,
                }
            )
    table = pd.DataFrame(rows)
    return table.sort_values(
        ["machine_type", "machine_id", "label", "clip"], ignore_index=True
    )


class DcaseAudio:
    """Audio source over a DCASE 2022 archive (mono 16-bit files)."""

    def __init__(self, archive: Path, table: pd.DataFrame) -> None:
        self.archive = archive
        self.table = table.reset_index(drop=True)
        self.fingerprint = f"dcase2022:{archive.name}:{archive.stat().st_size}"
        self._array: np.ndarray | None = None
        self._rows = {
            tuple(r): n for n, r in enumerate(self.table[list(FILE_COLUMNS)].to_numpy())
        }

    def index(self, machine_type: str, snr: int) -> pd.DataFrame:
        """File columns of the rows of :meth:`array`."""
        return self.table[list(FILE_COLUMNS)]

    def array(self, machine_type: str, snr: int) -> np.ndarray:
        """Int16 waveforms of every file of the archive, read once."""
        if self._array is None:
            with zipfile.ZipFile(self.archive) as zf:
                waves = []
                for member in self.table["member"]:
                    with zf.open(member) as f:
                        samples, rate = sf.read(f, dtype="int16")
                    if rate != 16000 or samples.ndim != 1:
                        raise ValueError(f"{member}: unexpected format")
                    waves.append(samples)
            self._array = np.stack(waves)
        return self._array

    def load(self, machine_type: str, snr: int, files: pd.DataFrame) -> np.ndarray:
        """Waveforms of the given files, in order."""
        order = [self._rows[tuple(r)] for r in files[list(FILE_COLUMNS)].to_numpy()]
        return np.asarray(self.array(machine_type, snr)[order])


def section_metrics(scores: pd.DataFrame, max_fpr: float) -> pd.DataFrame:
    """AUC (source), AUC (target) and pAUC per section, as ``01_test.py``.

    AUC of a domain: normal clips of that domain and all anomalous clips of the
    section; pAUC: all clips of the section.

    Args:
        scores: Test rows with ``machine_id``, ``domain``, ``y`` and ``score``.
        max_fpr: pAUC bound.

    Returns:
        One row per section, metrics in points (x 100).
    """
    rows = []
    for section, part in scores.groupby("machine_id", sort=True):
        row = {"section": section.removeprefix("section_")}
        for domain in ("source", "target"):
            kept = part[(part["domain"] == domain) | (part["y"] == 1)]
            row[f"auc_{domain}"] = 100 * auc(
                kept["y"].to_numpy(), kept["score"].to_numpy()
            )
        row["pauc"] = 100 * pauc(
            part["y"].to_numpy(), part["score"].to_numpy(), max_fpr
        )
        rows.append(row)
    return pd.DataFrame(rows)


def confusion(model: MobileNetV2Detector, clips: ClipSet) -> pd.DataFrame:
    """Confusion matrix of the auxiliary ID task on the given clips.

    Args:
        model: Trained detector.
        clips: Clips of known ID (normal test clips).

    Returns:
        Counts, rows = true ID, columns = predicted ID (argmax of the mean
        window probability).
    """
    predicted = model.id_probabilities(clips).mean(axis=1).argmax(axis=1)
    return pd.crosstab(
        pd.Categorical(clips.table["machine_id"], categories=model.classes),
        pd.Categorical([model.classes[k] for k in predicted], categories=model.classes),
        rownames=["true"],
        colnames=["predicted"],
        dropna=False,
    )


def compare(summary: pd.DataFrame) -> pd.DataFrame:
    """Applies the pre-registered selection and agreement rules.

    Args:
        summary: Mean and std over seeds per (machine type, section, metric).

    Returns:
        One row per retained published cell with our mean, std, difference and
        agreement flag.
    """
    published = pd.read_csv(PUBLISHED, dtype={"section": str})
    rows = []
    for metric in ("auc_source", "auc_target", "pauc"):
        for r in published[published[f"{metric}_std"] <= MAX_STD].itertuples():
            ours = summary[
                (summary["machine_type"] == r.machine_type)
                & (summary["section"] == r.section)
                & (summary["metric"] == metric)
            ]
            mean = float(ours["mean"].iloc[0]) if len(ours) else np.nan
            std = float(ours["std"].iloc[0]) if len(ours) else np.nan
            published_mean = getattr(r, f"{metric}_mean")
            rows.append(
                {
                    "machine_type": r.machine_type,
                    "section": r.section,
                    "metric": metric,
                    "published_mean": published_mean,
                    "published_std": getattr(r, f"{metric}_std"),
                    "our_mean": mean,
                    "our_std": std,
                    "difference": mean - published_mean,
                    "agrees": bool(abs(mean - published_mean) <= TOLERANCE),
                }
            )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    """Runs the fidelity check.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="MobileNetV2 fidelity check (DCASE 2022)."
    )
    parser.add_argument(
        "--types", nargs="+", default=["ToyCar", "ToyTrain", "fan", "slider", "valve"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="0.1 as the original; another value is a control, written apart.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    paths = load_paths()
    out_root = paths.output_root / "verification" / "mobilenetv2" / "dcase2022"
    if args.validation_split != 0.1:
        out_root = out_root.with_name(
            f"dcase2022_validation_split_{args.validation_split}"
        )
    out_root.mkdir(parents=True, exist_ok=True)
    setup_logging(out_root / "verification.log")
    dirty = git_is_dirty()
    if dirty and not args.allow_dirty:
        logger.error("Uncommitted changes: commit them or pass --allow-dirty")
        return 1
    config = copy.deepcopy(load_yaml(CONFIG_DIR / "models" / "mobilenetv2.yaml"))
    config["validation_split"] = args.validation_split  # original: 0.1
    max_fpr = float(
        load_yaml(CONFIG_DIR / "protocol.yaml")["evaluation"]["pauc_max_fpr"]
    )
    device = torch.device(args.device)
    commit = git_commit()

    rows = []
    for machine_type in args.types:
        archive = paths.data_root / "dcase2022" / f"dev_{machine_type}.zip"
        table = clip_table(archive)
        audio = DcaseAudio(archive, table)
        train = ClipSet(select(table, TRAIN), audio, paths.cache_root)
        test = ClipSet(select(table, TEST), audio, paths.cache_root)
        for seed in args.seeds:
            run_dir = out_root / machine_type / f"seed{seed}"
            done = run_dir / "result.json"
            if not done.exists():
                run_dir.mkdir(parents=True, exist_ok=True)
                started = time.time()
                seed_everything(seed)
                model = MobileNetV2Detector(config, seed, device)
                model.fit(train)
                scores = test.table[
                    ["machine_id", "domain", "label", "clip", "y"]
                ].copy()
                scores["score"] = model.score(test)
                scores.to_csv(run_dir / "scores.csv", index=False, lineterminator="\n")
                normal = test.subset(test.table["label"] == NORMAL)
                result = {
                    "machine_type": machine_type,
                    "seed": seed,
                    "config": config,
                    "git_commit": commit,
                    "git_dirty": dirty,
                    "history": model.history,
                    "steps": model.steps,
                    "sections": section_metrics(scores, max_fpr).to_dict("records"),
                    "confusion_test_normal": confusion(model, normal).to_dict(),
                    "environment": environment_info(),
                    "duration_s": time.time() - started,
                }
                done.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                logger.info(
                    "%s seed %d: %.0f s", machine_type, seed, result["duration_s"]
                )
            result = json.loads(done.read_text())
            for section in result["sections"]:
                for metric in ("auc_source", "auc_target", "pauc"):
                    rows.append(
                        {
                            "machine_type": machine_type,
                            "section": section["section"],
                            "metric": metric,
                            "seed": seed,
                            "value": section[metric],
                        }
                    )
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(out_root / "per_seed.csv", index=False, lineterminator="\n")
    summary = (
        per_seed.groupby(["machine_type", "section", "metric"])["value"]
        .agg(mean="mean", std=lambda v: v.std(ddof=1), n="size")
        .reset_index()
    )
    summary.to_csv(
        out_root / "summary.csv", index=False, lineterminator="\n", float_format="%.3f"
    )
    comparison = compare(summary)
    comparison.to_csv(
        out_root / "comparison.csv",
        index=False,
        lineterminator="\n",
        float_format="%.3f",
    )
    logger.info(
        "Comparison with the retained cells:\n%s",
        comparison.round(2).to_string(index=False),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
