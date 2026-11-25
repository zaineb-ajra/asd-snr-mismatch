"""Diagnostic of the OS-SCL model on the study data (not part of the study).

Retrains the internal-control model (main split, training at 0 dB, seed 0; the
training is deterministic, so this is the model of the internal controls), saves
the trained and EMA weights, and records:

- the per-class machine-ID accuracy on the training clips, for the trained and
  the EMA weights, every ``--every`` epochs;
- at the end, the same accuracy on the normal test clips at every SNR;
- the geometry of the embeddings per class (training clips and normal test clips
  at the training SNR): accuracy of a nearest-centroid classifier, compactness
  (mean cosine to the class centroid), nearest other centroid, cosine between
  the centroid and the ArcFace center of the class.

Usage::

    CUDA_VISIBLE_DEVICES=2 python -m noisemismatch.diagnose_osscl [--epochs 2]

Outputs: ``<output_root>/verification/osscl/diagnostic[-smoke]/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any

import numpy as np
import torch

from noisemismatch.config import CONFIG_DIR, load_paths, load_yaml
from noisemismatch.data.audio_cache import AudioCache, cache_dir
from noisemismatch.data.build_split import identity_sets
from noisemismatch.data.dataset import ClipSet, build_clip_table, select
from noisemismatch.data.identity import load_identity
from noisemismatch.data.prepare import SPLIT_PATH
from noisemismatch.data.split import NORMAL, TEST, TRAIN, load_split
from noisemismatch.models.osscl import OSSCLDetector, OSSCLNet
from noisemismatch.utils.logs import setup_logging
from noisemismatch.utils.provenance import environment_info, git_commit, git_is_dirty
from noisemismatch.utils.repro import seed_everything

logger = logging.getLogger(__name__)

TRAIN_SNR: int = 0


def geometry(
    classes: list[str],
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_test: np.ndarray,
    y_test: np.ndarray,
    centers: np.ndarray,
) -> list[dict[str, Any]]:
    """Per-class geometry of L2-normalized embeddings.

    Args:
        classes: Class names.
        z_train: Training embeddings ``(n, d)``.
        y_train: Their class indices.
        z_test: Embeddings of the normal test clips.
        y_test: Their class indices.
        centers: ArcFace class centers ``(n_classes, d)``.

    Returns:
        One record per class.
    """
    unit = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)  # noqa: E731
    centroids = unit(
        np.stack([z_train[y_train == k].mean(axis=0) for k in range(len(classes))])
    )
    centers = unit(centers)
    between = centroids @ centroids.T
    np.fill_diagonal(between, -np.inf)
    out = []
    for k, name in enumerate(classes):
        record: dict[str, Any] = {"class": name}
        for part, z, y in (("train", z_train, y_train), ("test", z_test, y_test)):
            own = z[y == k]
            sims = own @ centroids.T
            record[f"{part}_n"] = len(own)
            record[f"{part}_centroid_accuracy"] = float(
                (sims.argmax(axis=1) == k).mean()
            )
            record[f"{part}_arcface_accuracy"] = float(
                ((own @ centers.T).argmax(axis=1) == k).mean()
            )
            record[f"{part}_cosine_to_own_centroid"] = float(sims[:, k].mean())
        nearest = int(between[k].argmax())
        record["nearest_other_centroid"] = classes[nearest]
        record["cosine_to_nearest_other_centroid"] = float(between[k, nearest])
        record["cosine_centroid_to_arcface_center"] = float(centroids[k] @ centers[k])
        out.append(record)
    return out


def main(argv: list[str] | None = None) -> int:
    """Runs the diagnostic.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Diagnostic of OS-SCL.")
    parser.add_argument("--epochs", type=int, default=None, help="Smoke test.")
    parser.add_argument("--every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    paths = load_paths()
    name = "diagnostic" if args.epochs is None else "diagnostic-smoke"
    out = paths.output_root / "verification" / "osscl" / name
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / "diagnostic.log")
    dirty = git_is_dirty()
    if dirty and not args.allow_dirty:
        logger.error("Uncommitted changes: commit them or pass --allow-dirty")
        return 1
    protocol = load_yaml(CONFIG_DIR / "protocol.yaml")
    config = load_yaml(CONFIG_DIR / "models" / "osscl.yaml")
    if args.epochs is not None:
        config["epochs"] = args.epochs
    snrs = tuple(protocol["evaluation"]["test_snrs_db"])
    sets = identity_sets(load_identity(), protocol["dataset"]["excluded_machine_ids"])
    table = build_clip_table(
        load_split(SPLIT_PATH),
        sets["main"],
        snrs,
        protocol["evaluation"]["strict_variant"],
    )
    audio = AudioCache(cache_dir(paths.cache_root))
    train = ClipSet(select(table, TRAIN, (TRAIN_SNR,)), audio, paths.cache_root)
    test = ClipSet(select(table, TEST, snrs), audio, paths.cache_root)
    normal = test.subset(test.table["label"] == NORMAL)

    started = time.time()
    seed_everything(args.seed)
    model = OSSCLDetector(config, args.seed, torch.device(args.device))
    model.diagnostic_clips, model.diagnostic_every = train, args.every
    model.diagnostic_dir = out
    model.fit(train)
    torch.save(
        {
            "raw": model.net.state_dict(),
            "ema": model.ema.state_dict(),
            "classes": model.classes,
        },
        out / "weights.pt",
    )
    index = {c: k for k, c in enumerate(model.classes)}
    labels = lambda clips: np.array(  # noqa: E731
        [index[v] for v in model.class_labels(clips.table)]
    )
    result: dict[str, Any] = {
        "config": config,
        "seed": args.seed,
        "git_commit": git_commit(),
        "git_dirty": dirty,
        "history": model.history,
        "test_normal_accuracy": {},
        "geometry": {},
    }
    nets: dict[str, OSSCLNet] = {"raw": model.net, "ema": model.ema}
    at_train_snr = normal.subset(normal.table["snr_db"] == TRAIN_SNR)
    for name_, net in nets.items():
        result["test_normal_accuracy"][name_] = {}
        for snr in snrs:
            part = normal.subset(normal.table["snr_db"] == snr)
            per_class = model.class_accuracy(net, part)
            overall = float(
                (model.probabilities(net, part).argmax(axis=1) == labels(part)).mean()
            )
            result["test_normal_accuracy"][name_][str(snr)] = {
                "overall": overall,
                "per_class": per_class,
            }
        result["geometry"][name_] = geometry(
            model.classes,
            model.embeddings(net, train),
            labels(train),
            model.embeddings(net, at_train_snr),
            labels(at_train_snr),
            net.arcface.weight.detach().cpu().numpy(),
        )
    result["duration_s"] = time.time() - started
    result["environment"] = environment_info()
    (out / "diagnostic.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(
        "Overall ID accuracy on normal test clips at %+d dB: raw %.4f, EMA %.4f (%.0f s)",
        TRAIN_SNR,
        result["test_normal_accuracy"]["raw"][str(TRAIN_SNR)]["overall"],
        result["test_normal_accuracy"]["ema"][str(TRAIN_SNR)]["overall"],
        result["duration_s"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
