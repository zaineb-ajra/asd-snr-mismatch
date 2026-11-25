"""Applies the checklists of docs/phase6_checklist.md to the study runs.

Usage::

    python scripts/check_runs.py --stage A        # exit code 0 if every check passes

Writes ``<runs>/check_stage_<stage>.json``; ``<runs>`` is ``<output_root>/runs``
unless ``--runs-dir`` is given (tests).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from noisemismatch.config import REPO_ROOT, load_paths
from noisemismatch.data.build_split import identity_sets
from noisemismatch.data.dataset import build_clip_table, select
from noisemismatch.data.identity import load_identity
from noisemismatch.data.split import IDENTITY_COLUMNS, TEST, TRAIN, load_split
from noisemismatch.models.osscl import ema_decay
from noisemismatch.run import SPLIT_FILES

STAGES: dict[str, tuple[str, ...]] = {
    "A": ("ae", "mobilenetv2", "eat_knn", "wilkinghoff"),
    "B": ("osscl",),
}
DETERMINISTIC: tuple[str, ...] = ("eat_knn",)
KEY: list[str] = [*IDENTITY_COLUMNS, "test_snr_db"]
TOLERANCE: float = 1e-6  # the archived control AUCs have six decimals


def split_conditions(
    protocol: dict[str, Any],
) -> dict[str, list[tuple[str, tuple[int, ...]]]]:
    """Training conditions of each split: (name, training SNRs)."""
    conditions = [
        (c["name"], tuple(c["train_snrs_db"]))
        for c in protocol["training"]["conditions"]
    ]
    sens = set(protocol["sensitivity"]["snrs_db"])
    return {
        "main": conditions,
        "sensitivity": [c for c in conditions if len(c[1]) == 1 and c[1][0] in sens],
    }


def expected_runs(
    stage: str, protocol: dict[str, Any]
) -> list[tuple[str, str, str, int]]:
    """(model, split, condition, seed) of every run of a stage."""
    return [
        (model, split, name, seed)
        for model in STAGES[stage]
        for split, conditions in split_conditions(protocol).items()
        for name, _ in conditions
        for seed in protocol["training"]["seeds"]
    ]


class Checker:
    """Collects the failures of each check."""

    def __init__(self) -> None:
        self.failures: dict[str, list[str]] = {}
        self.counts: dict[str, int] = {}

    def check(self, name: str, ok: bool, message: str) -> bool:
        """Records one application of a check; returns ``ok``."""
        self.counts[name] = self.counts.get(name, 0) + 1
        if not ok:
            self.failures.setdefault(name, []).append(message)
        return ok


def committed_yaml(commit: str, path: str) -> dict[str, Any]:
    """Content of a YAML file of the repository at a commit (JSON types)."""
    text = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(json.dumps(yaml.safe_load(text)))


def reference_auc(model: str, output_root: Path) -> pd.DataFrame | None:
    """AUC per (machine type, ID, test SNR) of the Phase 4 control at 0 dB."""
    if model == "eat_knn":
        table = pd.read_csv(
            REPO_ROOT / "results" / "verification" / "eat_knn_controls.csv"
        )
        table = table[table["main"]]
        return table[["machine_type", "machine_id", "test_snr_db", "auc"]]
    path = (
        output_root
        / "verification"
        / model
        / "internal"
        / "0dB"
        / "seed0"
        / "controls.json"
    )
    if not path.exists():
        return None
    rows = []
    for group, record in json.loads(path.read_text())["groups"].items():
        for snr, per_snr in record["per_snr"].items():
            for s in per_snr["scores"]:
                rows.append(
                    {
                        "machine_type": s.get("machine_type", group),
                        "machine_id": s["machine_id"],
                        "test_snr_db": int(snr),
                        "auc": s["auc"],
                    }
                )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    """Runs the checks of one stage.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        0 if every check passes, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description="Checklist of the study runs.")
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--runs-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    paths = load_paths()
    runs = args.runs_dir or paths.output_root / "runs"
    commit = (runs / "PHASE6_COMMIT").read_text().strip()
    protocol = committed_yaml(commit, "configs/protocol.yaml")
    configs = {
        model: committed_yaml(commit, f"configs/models/{model}.yaml")
        for model in STAGES[args.stage]
    }
    sets = identity_sets(load_identity(), protocol["dataset"]["excluded_machine_ids"])
    expected: dict[str, dict[str, Any]] = {}
    for split in SPLIT_FILES:
        snrs = (
            tuple(protocol["evaluation"]["test_snrs_db"])
            if split == "main"
            else tuple(protocol["sensitivity"]["snrs_db"])
        )
        table = build_clip_table(
            load_split(SPLIT_FILES[split]),
            sets[split],
            snrs,
            protocol["evaluation"]["strict_variant"],
        )
        test = select(table, TEST, snrs)
        keys = set(
            map(tuple, test[[*IDENTITY_COLUMNS, "snr_db"]].astype(str).to_numpy())
        )
        train = set(
            map(
                tuple,
                select(table, TRAIN, snrs)[list(IDENTITY_COLUMNS)]
                .astype(str)
                .to_numpy(),
            )
        )
        n_ids = test.groupby(["machine_type", "machine_id"]).ngroups
        expected[split] = {"keys": keys, "train": train, "n_metrics": n_ids * len(snrs)}

    c = Checker()
    loaded: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for model, split, condition, seed in expected_runs(args.stage, protocol):
        run_id = f"{split}/{model}/{condition}/seed{seed}"
        run_dir = runs / split / model / condition / f"seed{seed}"
        files = {
            n: run_dir / n for n in ("DONE", "result.json", "scores.csv", "run.log")
        }
        if not c.check(
            "C1", all(p.exists() for p in files.values()), f"{run_id}: missing files"
        ):
            continue
        result = json.loads(files["result.json"].read_text())
        c.check(
            "C2",
            result.get("git_commit") == commit
            and result.get("git_dirty") is False
            and result.get("model_config") == configs[model]
            and result.get("protocol") == protocol,
            f"{run_id}: provenance (commit {result.get('git_commit')}, dirty {result.get('git_dirty')})",
        )
        scores = pd.read_csv(files["scores.csv"], dtype={k: str for k in KEY})
        keys = list(map(tuple, scores[KEY].astype(str).to_numpy()))
        identities = set(
            map(tuple, scores[list(IDENTITY_COLUMNS)].astype(str).to_numpy())
        )
        c.check(
            "C3",
            len(keys) == len(expected[split]["keys"])
            and set(keys) == expected[split]["keys"]
            and not identities & expected[split]["train"],
            f"{run_id}: test rows {len(keys)} (expected {len(expected[split]['keys'])})",
        )
        c.check(
            "C4",
            bool(np.isfinite(scores["score"]).all()),
            f"{run_id}: non-finite scores",
        )
        metrics = pd.DataFrame(result["metrics"])
        c.check(
            "C5",
            len(metrics) == expected[split]["n_metrics"]
            and bool(
                metrics[["auc", "pauc"]].apply(lambda v: v.between(0, 1)).all().all()
            ),
            f"{run_id}: metrics ({len(metrics)} rows)",
        )
        log = files["run.log"].read_text(errors="replace")
        c.check(
            "C7",
            "Traceback" not in log and " ERROR " not in log,
            f"{run_id}: error in run.log",
        )
        loaded[(model, split, condition, seed)] = {
            "scores": scores.sort_values(KEY)["score"].to_numpy(),
            "metrics": metrics,
            "diagnostics": result.get("model_diagnostics", {}),
        }

    conditions = split_conditions(protocol)
    seeds = protocol["training"]["seeds"]
    for model in STAGES[args.stage]:
        for split, conds in conditions.items():
            for name, train_snrs in conds:
                runs_ = [loaded.get((model, split, name, s)) for s in seeds]
                if any(r is None for r in runs_):
                    continue
                same = all(
                    np.array_equal(runs_[0]["scores"], r["scores"]) for r in runs_[1:]
                )
                different = all(
                    not np.array_equal(runs_[i]["scores"], runs_[j]["scores"])
                    for i in range(len(runs_))
                    for j in range(i + 1, len(runs_))
                )
                c.check(
                    "C6",
                    same if model in DETERMINISTIC else different,
                    f"{model}/{split}/{name}: seeds {'differ' if model in DETERMINISTIC else 'coincide'}",
                )
                if len(train_snrs) == 1:
                    for seed, run in zip(seeds, runs_, strict=True):
                        m = run["metrics"]
                        matched = m[m["test_snr_db"] == train_snrs[0]]["auc"].mean()
                        c.check(
                            "C10",
                            bool(matched > 0.5),
                            f"{model}/{split}/{name}/seed{seed}: matched mean AUC {matched:.3f}",
                        )
        if model not in DETERMINISTIC:
            for seed in seeds:
                steps = {}
                for name, _ in conditions["main"]:
                    run = loaded.get((model, "main", name, seed))
                    if run is not None:
                        steps[name] = {
                            g: d.get("steps") for g, d in run["diagnostics"].items()
                        }
                values = list(steps.values())
                c.check(
                    "C9",
                    len(values) == len(conditions["main"])
                    and bool(values[0])
                    and all(v == values[0] for v in values[1:]),
                    f"{model}/seed{seed}: steps differ between conditions or are missing",
                )
        reference = reference_auc(model, paths.output_root)
        name = "S2" if args.stage == "B" else "C8"
        if model in ("mobilenetv2", "eat_knn", "osscl"):
            run = loaded.get((model, "main", "0dB", 0))
            ok = False
            if run is not None and reference is not None:
                merged = run["metrics"].merge(
                    reference,
                    on=["machine_type", "machine_id", "test_snr_db"],
                    suffixes=("", "_ref"),
                )
                ok = len(merged) == len(run["metrics"]) and bool(
                    ((merged["auc"] - merged["auc_ref"]).abs() <= TOLERANCE).all()
                )
            c.check(
                name,
                ok,
                f"{model}: main/0dB/seed0 does not reproduce the Phase 4 control",
            )
        if args.stage == "B":
            fraction = float(configs["osscl"]["ema_window_fraction"])
            for key, run in loaded.items():
                for group, d in run["diagnostics"].items():
                    raw, ema = d["train_id_accuracy_raw"], d["train_id_accuracy_ema"]
                    worst = max(abs(raw[k] - ema[k]) for k in raw)
                    c.check(
                        "S1",
                        abs(d["ema_decay"] - ema_decay(fraction, d["steps"])) < 1e-12
                        and worst <= 0.01,
                        f"{'/'.join(map(str, key))} [{group}]: EMA safeguard (largest gap {worst:.3f})",
                    )
                c.check(
                    "S1",
                    bool(run["diagnostics"]),
                    f"{'/'.join(map(str, key))}: no safeguard recorded",
                )

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    report = {
        "stage": args.stage,
        "phase6_commit": commit,
        "head_at_check": head,
        "checks": {
            name: {
                "n": n,
                "failed": len(c.failures.get(name, [])),
                "failures": c.failures.get(name, []),
            }
            for name, n in sorted(c.counts.items())
        },
        "passed": not c.failures,
    }
    (runs / f"check_stage_{args.stage}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    for name, info in report["checks"].items():
        print(f"{name}: {info['n'] - info['failed']} of {info['n']} pass")
        for message in info["failures"][:10]:
            print(f"    FAIL {message}")
    print("STAGE", args.stage, "PASSED" if report["passed"] else "FAILED")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
