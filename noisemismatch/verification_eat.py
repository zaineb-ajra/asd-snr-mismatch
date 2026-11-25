"""Code validation of EAT + kNN (not part of the study).

Two checks, defined in docs/validation.md ("EAT + kNN") before they were run:

- ``fidelity``: the study pipeline (channel-0 audio cache, vendored encoder,
  batched passes, cached embeddings) against the official inference path of the
  model card (``scripts/eat_reference.py``, run in a separate environment with
  ``transformers``), on 90 clips; criterion fixed in docs/validation.md.
- ``controls``: on the study data (main split, reference set at 0 dB):
  determinism of the embeddings, self-distance of training clips, AUC and score
  quartiles per machine ID and test SNR, and the sensitivity analysis of the
  protocol (pooling x K) for this condition.

Usage::

    CUDA_VISIBLE_DEVICES=1 python -m noisemismatch.verification_eat fidelity --reference-python <venv>/bin/python
    CUDA_VISIBLE_DEVICES=1 python -m noisemismatch.verification_eat controls

Outputs: ``<output_root>/verification/eat_knn/{fidelity,internal/0dB}/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from noisemismatch.config import CONFIG_DIR, REPO_ROOT, load_paths, load_yaml
from noisemismatch.data.audio_cache import AudioCache, cache_dir
from noisemismatch.data.build_split import identity_sets
from noisemismatch.data.dataset import ClipSet, build_clip_table, select
from noisemismatch.data.dcase2020 import mono_wav_bytes
from noisemismatch.data.identity import load_identity
from noisemismatch.data.prepare import SPLIT_PATH
from noisemismatch.data.split import TEST, TRAIN, load_split
from noisemismatch.eval.metrics import auc, pauc
from noisemismatch.features.cache import CHUNK, FeatureCache
from noisemismatch.features.eat import (
    POOLINGS,
    EatEmbedding,
    checkpoint_dir,
    filterbank,
)
from noisemismatch.models.eat_knn import EatKnnDetector
from noisemismatch.utils.logs import setup_logging
from noisemismatch.utils.provenance import environment_info, git_commit, git_is_dirty
from noisemismatch.utils.repro import seed_everything

logger = logging.getLogger(__name__)

FIDELITY_SEED: int = 2026
RELATIVE_TOLERANCE: float = 1e-4
MIN_COSINE: float = 0.99999
CONTROL_SNR: int = 0


def study_table(protocol: dict[str, Any]) -> pd.DataFrame:
    """Clip table of the main split."""
    snrs = tuple(protocol["evaluation"]["test_snrs_db"])
    sets = identity_sets(load_identity(), protocol["dataset"]["excluded_machine_ids"])
    return build_clip_table(
        load_split(SPLIT_PATH),
        sets["main"],
        snrs,
        protocol["evaluation"]["strict_variant"],
    )


def fidelity_clips(table: pd.DataFrame, seed: int = FIDELITY_SEED) -> pd.DataFrame:
    """One normal and one anomalous test clip per machine ID and SNR.

    Args:
        table: Clip table of the main split.
        seed: Seed of the draw.

    Returns:
        The drawn rows, in group order.
    """
    rng = np.random.default_rng(seed)
    test = select(table, TEST)
    rows = []
    for _, group in test.groupby(["machine_type", "machine_id", "snr_db", "label"]):
        rows.append(group.iloc[int(rng.integers(len(group)))])
    return pd.DataFrame(rows).reset_index(drop=True)


def compare(ours: dict[str, np.ndarray], ref: dict[str, np.ndarray]) -> pd.DataFrame:
    """Per-clip relative errors and cosine similarities (criterion of validation.md).

    Args:
        ours: ``fbank``, ``time_mean``, ``cls`` arrays of the study pipeline.
        ref: The same arrays from the reference.

    Returns:
        One row per clip with the errors and a ``passes`` flag.
    """
    out = pd.DataFrame(index=range(len(ref["cls"])))
    passes = np.ones(len(out), dtype=bool)
    for name in ("fbank", "time_mean", "cls"):
        a = ours[name].reshape(len(out), -1).astype(np.float64)
        b = ref[name].reshape(len(out), -1).astype(np.float64)
        rel = np.abs(a - b).max(axis=1) / np.abs(b).max(axis=1)
        out[f"{name}_rel_error"] = rel
        passes &= rel <= RELATIVE_TOLERANCE
        if name != "fbank":
            cos = (
                (a * b).sum(axis=1)
                / np.linalg.norm(a, axis=1)
                / np.linalg.norm(b, axis=1)
            )
            out[f"{name}_cosine"] = cos
            passes &= cos >= MIN_COSINE
    out["passes"] = passes
    return out


def run_fidelity(
    args: argparse.Namespace, out: Path, protocol: dict[str, Any]
) -> tuple[dict[str, Any], int]:
    """Fidelity check (see module docstring)."""
    paths = load_paths()
    config = load_yaml(CONFIG_DIR / "models" / "eat_knn.yaml")
    device = torch.device(args.device)
    clips = fidelity_clips(study_table(protocol))
    audio = AudioCache(cache_dir(paths.cache_root))
    clip_set = ClipSet(clips, audio, paths.cache_root)
    extractor = EatEmbedding(
        device=device, **config["checkpoint"], **config["features"]
    )
    embeddings = clip_set.features(extractor)  # study cache
    waves = clip_set.waveforms()
    c = config["features"]
    with torch.no_grad():
        fbank = filterbank(
            torch.from_numpy(waves).to(device),
            int(c["num_mel_bins"]),
            float(c["frame_shift_ms"]),
            int(c["target_length"]),
            float(c["norm_mean"]),
            float(c["norm_std"]),
        )[:, 0]
    ours = {
        "fbank": fbank.cpu().numpy(),
        "time_mean": embeddings[:, POOLINGS["time_mean"]],
        "cls": embeddings[:, POOLINGS["cls"]],
    }
    wav_dir = out / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    names = []
    raw = np.round(waves * 32768.0).astype(np.int16)
    for k, row in clips.iterrows():
        path = wav_dir / (
            f"{k:03d}_{row['machine_type']}_{row['machine_id']}_{row['label']}"
            f"_{row['clip']}_{int(row['snr_db']):+d}dB.wav"
        )
        path.write_bytes(mono_wav_bytes(raw[k]))
        names.append(str(path))
    wav_list = out / "wav_list.txt"
    wav_list.write_text("\n".join(names) + "\n", encoding="utf-8")
    snapshot = checkpoint_dir(
        paths.data_root, config["checkpoint"]["repo"], config["checkpoint"]["revision"]
    )
    ref_path = out / "reference.npz"
    env = {**os.environ, "HF_HOME": str(paths.cache_root / "hf-home")}
    subprocess.run(
        [
            args.reference_python,
            str(REPO_ROOT / "scripts" / "eat_reference.py"),
            str(snapshot),
            str(wav_list),
            str(ref_path),
        ],
        check=True,
        env=env,
    )
    ref = np.load(ref_path)
    table = compare(ours, {k: ref[k] for k in ("fbank", "time_mean", "cls")})
    result_table = pd.concat(
        [clips[["machine_type", "machine_id", "label", "clip", "snr_db"]], table],
        axis=1,
    )
    result_table.to_csv(out / "fidelity.csv", index=False, lineterminator="\n")
    summary = {
        "n_clips": len(table),
        "n_pass": int(table["passes"].sum()),
        "passes": bool(table["passes"].all()),
        "max_rel_error": {
            n: float(table[f"{n}_rel_error"].max())
            for n in ("fbank", "time_mean", "cls")
        },
        "min_cosine": {
            n: float(table[f"{n}_cosine"].min()) for n in ("time_mean", "cls")
        },
        "tolerance": {"relative": RELATIVE_TOLERANCE, "min_cosine": MIN_COSINE},
        "reference": {
            "transformers": str(ref["transformers_version"]),
            "torch": str(ref["torch_version"]),
        },
    }
    logger.info("Fidelity: %s", summary)
    return summary, 0 if summary["passes"] else 1


def run_controls(
    args: argparse.Namespace, out: Path, protocol: dict[str, Any]
) -> tuple[dict[str, Any], int]:
    """Internal controls (see module docstring)."""
    paths = load_paths()
    config = load_yaml(CONFIG_DIR / "models" / "eat_knn.yaml")
    device = torch.device(args.device)
    table = study_table(protocol)
    audio = AudioCache(cache_dir(paths.cache_root))
    snrs = tuple(protocol["evaluation"]["test_snrs_db"])
    max_fpr = float(protocol["evaluation"]["pauc_max_fpr"])
    train = ClipSet(select(table, TRAIN, (CONTROL_SNR,)), audio, paths.cache_root)
    test = ClipSet(select(table, TEST, snrs), audio, paths.cache_root)
    test_groups = dict(test.groups("machine_id"))
    rows, self_distance = [], {}
    for key, train_group in train.groups("machine_id"):
        model = EatKnnDetector(config, 0, device)
        model.fit(train_group)
        own = model.score_variants(train_group, (("time_mean", 1),))
        self_distance["/".join(key)] = float(np.abs(own[("time_mean", 1)]).max())
        for snr in snrs:
            group = test_groups[key].subset(test_groups[key].table["snr_db"] == snr)
            y = group.table["y"].to_numpy()
            for (pooling, k), scores in model.score_variants(group).items():
                record = {
                    "machine_type": key[0],
                    "machine_id": key[1],
                    "test_snr_db": snr,
                    "pooling": pooling,
                    "k": k,
                    "main": (pooling, k) == (config["pooling"], int(config["k"])),
                    "auc": auc(y, scores),
                    "pauc": pauc(y, scores, max_fpr),
                }
                for label, name in ((0, "normal"), (1, "anomalous")):
                    q = np.quantile(scores[y == label], [0.25, 0.5, 0.75])
                    record[f"{name}_quartiles"] = [float(v) for v in q]
                rows.append(record)
        logger.info("%s: done", key)
    results = pd.DataFrame(rows)
    results.to_json(out / "controls.json", orient="records", indent=2)
    # determinism: one (machine type, SNR) array recomputed with a fresh extractor
    machine_type = "valve"
    fresh = EatEmbedding(device=device, **config["checkpoint"], **config["features"])
    cached = FeatureCache(paths.cache_root, fresh, audio)
    files = audio.index(machine_type, CONTROL_SNR)
    stored = cached.get(machine_type, CONTROL_SNR, files)
    waveforms = audio.array(machine_type, CONTROL_SNR)
    again = np.concatenate(
        [
            fresh.batch(
                np.asarray(waveforms[s : s + CHUNK], dtype=np.float32) / 32768.0
            )
            for s in range(0, len(waveforms), CHUNK)
        ]
    )
    deterministic = bool(np.array_equal(stored, again))
    summary = {
        "condition": f"reference set at {CONTROL_SNR:+d} dB",
        "self_distance_max": max(self_distance.values()),
        "self_distance_per_id": self_distance,
        "deterministic": deterministic,
        "determinism_array": f"{machine_type} {CONTROL_SNR:+d} dB ({len(stored)} clips)",
    }
    logger.info("Controls: %s", summary)
    ok = deterministic and summary["self_distance_max"] <= 1e-6
    return summary, 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    """Runs one check.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        0 if the check passes.
    """
    parser = argparse.ArgumentParser(description="EAT + kNN code validation.")
    parser.add_argument("check", choices=["fidelity", "controls"])
    parser.add_argument("--reference-python", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    paths = load_paths()
    sub = "fidelity" if args.check == "fidelity" else f"internal/{CONTROL_SNR}dB"
    out = paths.output_root / "verification" / "eat_knn" / sub
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / f"{args.check}.log")
    dirty = git_is_dirty()
    if dirty and not args.allow_dirty:
        logger.error("Uncommitted changes: commit them or pass --allow-dirty")
        return 1
    if args.check == "fidelity" and not args.reference_python:
        parser.error("fidelity needs --reference-python")
    seed_everything(0)
    protocol = load_yaml(CONFIG_DIR / "protocol.yaml")
    run = run_fidelity if args.check == "fidelity" else run_controls
    summary, code = run(args, out, protocol)
    summary.update(
        {
            "git_commit": git_commit(),
            "git_dirty": dirty,
            "config": load_yaml(CONFIG_DIR / "models" / "eat_knn.yaml"),
            "environment": environment_info(),
            "command": sys.argv,
        }
    )
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
