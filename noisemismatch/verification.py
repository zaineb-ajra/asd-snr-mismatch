"""Code validation of a model against published results (not part of the study).

Trains a model on a few (machine type, machine ID, SNR) pairs, trained and tested
at the same SNR, with two splits:

- ``study``: the main split of the study (content-based identities);
- ``sensitivity``: the 2x2 sensitivity split (pairs at 0 or +6 dB only), same
  draw as the study split but on all clips identifiable between 0 and +6 dB,
  hence almost without exclusion;
- ``original``: the split of the MIMII baseline code, the only one comparable
  with published AUCs: per SNR folder, file names sorted, the first N normal
  files (N = number of anomalous files) and all anomalous files in the test set,
  the other normal files for training. File names are taken at face value and all
  files are used, as in the baseline.

Outputs go to ``<output_root>/verification/<model>/`` and never mix with the runs of
the study. Usage::

    CUDA_VISIBLE_DEVICES=1 python -m noisemismatch.verification --model ae
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from noisemismatch.config import CONFIG_DIR, load_paths, load_yaml
from noisemismatch.data.audio_cache import (
    AudioCache,
    cache_dir,
    member_name,
    read_channel,
)
from noisemismatch.data.build_split import SENSITIVITY_SPLIT_PATH, identity_sets
from noisemismatch.data.dataset import build_clip_table
from noisemismatch.data.download import DOWNLOAD_SUBDIR, load_manifest
from noisemismatch.data.identity import load_identity
from noisemismatch.data.prepare import SPLIT_PATH, index_path, load_index
from noisemismatch.data.split import (
    ANOMALOUS,
    FILE_COLUMNS,
    NORMAL,
    TEST,
    TRAIN,
    load_split,
)
from noisemismatch.run import DONE, RunSpec, execute, write_run
from noisemismatch.utils.logs import setup_logging
from noisemismatch.utils.provenance import environment_info, git_commit, git_is_dirty

logger = logging.getLogger(__name__)

# (machine type, machine ID, SNR, AUC published in Table 3 of the MIMII paper).
PAIRS: list[tuple[str, str, int, float]] = [
    ("fan", "id_02", 6, 0.99),  # easy: near the ceiling
    ("valve", "id_04", -6, 0.50),  # lowest valve cell among IDs of the study
    ("pump", "id_02", 6, 0.45),  # lowest cell of the table among released IDs
]
SEEDS: tuple[int, ...] = (0, 1, 2)


class ArchiveAudio:
    """Channel 0 of all files of one machine ID, read from its archive.

    Args:
        downloads_dir: Directory holding the archives.
        machine_type: Machine type.
        snr: SNR in dB.
        machine_id: Machine ID.
        files: File columns of the files of this machine ID.
    """

    def __init__(
        self,
        downloads_dir: Path,
        machine_type: str,
        snr: int,
        machine_id: str,
        files: pd.DataFrame,
    ) -> None:
        self.archive = f"{snr}_dB_{machine_type}.zip"
        self.path = downloads_dir / self.archive
        self.key = (machine_type, snr)
        self.files = files.reset_index(drop=True)
        md5 = {a.filename: a.md5 for a in load_manifest()}[self.archive]
        self.fingerprint = f"archive-ch0:{self.archive}:{md5}:{machine_id}:{len(files)}"
        self._array: np.ndarray | None = None
        self._lock = threading.Lock()

    def index(self, machine_type: str, snr: int) -> pd.DataFrame:
        """File columns of the rows of :meth:`array`."""
        assert (machine_type, snr) == self.key
        return self.files

    def array(self, machine_type: str, snr: int) -> np.ndarray:
        """Int16 channel-0 waveforms of all files, read once."""
        assert (machine_type, snr) == self.key
        with self._lock:
            if self._array is None:
                with zipfile.ZipFile(self.path) as zf:
                    self._array = np.stack(
                        [
                            read_channel(zf, member_name(machine_type, *row))
                            for row in self.files[list(FILE_COLUMNS)].to_numpy()
                        ]
                    )
        return self._array

    def load(self, machine_type: str, snr: int, files: pd.DataFrame) -> np.ndarray:
        """Waveforms of the given files, in order."""
        rows = {
            tuple(r): n for n, r in enumerate(self.files[list(FILE_COLUMNS)].to_numpy())
        }
        order = [rows[tuple(r)] for r in files[list(FILE_COLUMNS)].to_numpy()]
        return np.asarray(self.array(machine_type, snr)[order])


def original_split_table(
    index: pd.DataFrame, machine_type: str, machine_id: str, snr: int
) -> pd.DataFrame:
    """Clip table of the MIMII baseline split for one machine ID and SNR.

    Args:
        index: File index of the archives.
        machine_type: Machine type.
        machine_id: Machine ID.
        snr: SNR in dB.

    Returns:
        A clip table (same columns as the study's) where identities are file
        names at this SNR.
    """
    mask = (
        (index["machine_type"] == machine_type)
        & (index["machine_id"] == machine_id)
        & (index["snr_db"] == snr)
    )
    files = index[mask]
    normal = files[files["label"] == NORMAL].sort_values("clip")
    anomalous = files[files["label"] == ANOMALOUS].sort_values("clip")
    n_test = len(anomalous)
    table = pd.concat([normal, anomalous], ignore_index=True)[
        ["machine_type", "machine_id", "label", "clip"]
    ]
    table["snr_db"] = snr
    table["file_machine_id"] = table["machine_id"]
    table["file_label"] = table["label"]
    table["file_clip"] = table["clip"]
    table["split"] = TRAIN
    table.loc[: n_test - 1, "split"] = TEST  # first N normal files in sorted order
    table.loc[table["label"] == ANOMALOUS, "split"] = TEST
    table["y"] = (table["label"] == ANOMALOUS).astype(int)
    table["strict_exclude"] = False
    return table.sort_values(
        ["machine_type", "machine_id", "label", "clip"], ignore_index=True
    )


def restrict(table: pd.DataFrame, machine_type: str, machine_id: str) -> pd.DataFrame:
    """Rows of a clip table for one machine ID, reindexed."""
    mask = (table["machine_type"] == machine_type) & (table["machine_id"] == machine_id)
    return table[mask].reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    """Runs the validation of one model on :data:`PAIRS`.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Code validation against published AUCs."
    )
    parser.add_argument("--model", default="ae")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)

    import noisemismatch.models  # noqa: F401  (registers the models)

    paths = load_paths()
    out_root = paths.output_root / "verification" / args.model
    out_root.mkdir(parents=True, exist_ok=True)
    setup_logging(out_root / "verification.log")
    dirty = git_is_dirty()
    if dirty and not args.allow_dirty:
        logger.error("Uncommitted changes: commit them or pass --allow-dirty")
        return 1
    provenance = {"git_commit": git_commit(), "git_dirty": dirty}
    protocol = load_yaml(CONFIG_DIR / "protocol.yaml")
    base_config = load_yaml(CONFIG_DIR / "models" / f"{args.model}.yaml")
    max_fpr = float(protocol["evaluation"]["pauc_max_fpr"])
    device = torch.device(args.device)

    sets = identity_sets(load_identity(), protocol["dataset"]["excluded_machine_ids"])
    strict = protocol["evaluation"]["strict_variant"]
    study_table = build_clip_table(
        load_split(SPLIT_PATH), sets["main"], (0, 6, -6), strict
    )
    sensitivity_snrs = tuple(protocol["sensitivity"]["snrs_db"])
    sensitivity_table = build_clip_table(
        load_split(SENSITIVITY_SPLIT_PATH),
        sets["sensitivity"],
        sensitivity_snrs,
        strict,
    )
    index = load_index(index_path())
    audio = AudioCache(cache_dir(paths.cache_root))
    downloads = paths.data_root / DOWNLOAD_SUBDIR

    rows = []
    for machine_type, machine_id, snr, published in PAIRS:
        pair = f"{machine_type}_{machine_id}_{snr:+d}dB"
        original = original_split_table(index, machine_type, machine_id, snr)
        archive_audio = ArchiveAudio(
            downloads, machine_type, snr, machine_id, original[list(FILE_COLUMNS)]
        )

        kinds = [("study", restrict(study_table, machine_type, machine_id), audio)]
        if snr in sensitivity_snrs:
            table = restrict(sensitivity_table, machine_type, machine_id)
            kinds.append(("sensitivity", table, audio))
        kinds.append(("original", original, archive_audio))
        for split_kind, table, source in kinds:
            for fraction in (0.1, 0.0):
                config = copy.deepcopy(base_config)
                config["validation_split"] = fraction
                for seed in SEEDS:
                    spec = RunSpec(
                        args.model,
                        f"verification-{split_kind}",
                        f"{snr:+d}dB",
                        (snr,),
                        (snr,),
                        seed,
                    )
                    run_dir = (
                        out_root
                        / split_kind
                        / pair
                        / f"validation_split_{fraction}"
                        / f"seed{seed}"
                    )
                    if not (run_dir / DONE).exists():
                        run_dir.mkdir(parents=True, exist_ok=True)
                        started = time.time()
                        scores, result = execute(
                            spec,
                            table,
                            source,
                            paths.cache_root,
                            config,
                            device,
                            max_fpr,
                            provenance,
                        )
                        result.update(
                            {
                                "verification": {
                                    "split": split_kind,
                                    "pair": pair,
                                    "published_auc": published,
                                    "validation_split": fraction,
                                },
                                "environment": environment_info(),
                                "duration_s": time.time() - started,
                                "finished_utc": pd.Timestamp.now(tz="UTC").isoformat(
                                    timespec="seconds"
                                ),
                            }
                        )
                        write_run(run_dir, scores, result)
                    result = json.loads((run_dir / "result.json").read_text())
                    (metric,) = result["metrics"]
                    rows.append(
                        {
                            "split": split_kind,
                            "pair": pair,
                            "validation_split": fraction,
                            "seed": seed,
                            "n_train_clips": int(
                                (
                                    (table["split"] == TRAIN) & (table["snr_db"] == snr)
                                ).sum()
                            ),
                            "n_test_normal": metric["n_test_normal"],
                            "n_test_anomalous": metric["n_test_anomalous"],
                            "auc": metric["auc"],
                            "pauc": metric["pauc"],
                            "published_auc": published,
                            "duration_s": result["duration_s"],
                        }
                    )
                    logger.info(
                        "%s %s vs=%.1f seed %d: AUC %.4f",
                        split_kind,
                        pair,
                        fraction,
                        seed,
                        metric["auc"],
                    )
    summary = pd.DataFrame(rows)
    summary.to_csv(
        out_root / "summary.csv", index=False, lineterminator="\n", float_format="%.5f"
    )
    logger.info("Summary written to %s", out_root / "summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
