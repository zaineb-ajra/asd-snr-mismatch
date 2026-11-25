"""Tables of the MobileNetV2 code validation (not part of the study).

Usage::

    python scripts/mobilenetv2_report.py [--condition 0dB --seed 0] > tables.md

Reads the outputs of ``python -m noisemismatch.verification_dcase2022`` (fidelity
check on DCASE 2022) and ``python -m noisemismatch.controls --model mobilenetv2``
(internal controls on the study data), under
``<output_root>/verification/mobilenetv2/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from noisemismatch.config import load_paths

METRICS = {"auc_source": "AUC (source)", "auc_target": "AUC (target)", "pauc": "pAUC"}


def markdown(table: pd.DataFrame) -> str:
    """Renders a DataFrame as a GitHub-flavored Markdown table."""
    header = "| " + " | ".join(str(c) for c in table.columns) + " |"
    rule = "|" + "|".join("---" for _ in table.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.values]
    return "\n".join([header, rule, *rows])


def fidelity(root: Path) -> list[str]:
    """Comparison of the retained cells and ID accuracy on DCASE 2022."""
    out = ["## Fidelity check (DCASE 2022 Task 2 development set)", ""]
    comparison_path = root / "comparison.csv"
    if not comparison_path.exists():
        return [*out, "Not run.", ""]
    comparison = pd.read_csv(comparison_path, dtype={"section": str})
    n_agree = int(comparison["agrees"].sum())
    comparison = comparison.assign(
        metric=comparison["metric"].map(METRICS),
        published=[
            f"{m:.2f} ± {s:.2f}"
            for m, s in zip(
                comparison["published_mean"], comparison["published_std"], strict=True
            )
        ],
        ours=[
            f"{m:.2f} ± {s:.2f}"
            for m, s in zip(comparison["our_mean"], comparison["our_std"], strict=True)
        ],
        difference=comparison["difference"].map("{:+.2f}".format),
        agrees=comparison["agrees"].map({True: "yes", False: "no"}),
    )
    out += [
        "### Retained cells (published std ≤ 3.0; agreement: |difference| ≤ 3.0)",
        "",
        markdown(
            comparison[
                [
                    "metric",
                    "machine_type",
                    "section",
                    "published",
                    "ours",
                    "difference",
                    "agrees",
                ]
            ]
        ),
        "",
        f"{n_agree} of {len(comparison)} retained cells agree.",
        "",
    ]
    rows = []
    for result_path in sorted(root.glob("*/seed*/result.json")):
        result = json.loads(result_path.read_text())
        matrix = pd.DataFrame(result["confusion_test_normal"])
        rows.append(
            {
                "machine_type": result["machine_type"],
                "seed": result["seed"],
                "accuracy": np.trace(matrix.to_numpy()) / matrix.to_numpy().sum(),
                "duration_min": result["duration_s"] / 60,
            }
        )
    if rows:
        accuracy = (
            pd.DataFrame(rows)
            .groupby("machine_type")
            .agg(
                seeds=("seed", "size"),
                accuracy=(
                    "accuracy",
                    lambda v: f"{v.mean():.3f} ± {v.std(ddof=1):.3f}",
                ),
                minutes=("duration_min", lambda v: f"{v.mean():.1f}"),
            )
            .reset_index()
        )
        out += [
            "### Section (ID) classification accuracy on the normal test clips",
            "",
            markdown(accuracy),
            "",
        ]
    return out


def validation_split_control(root: Path) -> list[str]:
    """ToyCar seed 0 with validation_split 0.1 (original) and 0.0 (control)."""
    out = ["## Control run: ToyCar, seed 0, validation_split 0.1 vs 0.0", ""]
    paths = {
        "0.1": root / "dcase2022" / "ToyCar" / "seed0" / "result.json",
        "0.0": root
        / "dcase2022_validation_split_0.0"
        / "ToyCar"
        / "seed0"
        / "result.json",
    }
    if not all(path.exists() for path in paths.values()):
        return [*out, "Not run.", ""]
    rows = []
    for split, path in paths.items():
        result = json.loads(path.read_text())
        matrix = pd.DataFrame(result["confusion_test_normal"])  # rows true
        for section in result["sections"]:
            name = f"section_{section['section']}"
            rows.append(
                {
                    "section": section["section"],
                    "validation_split": split,
                    "ID accuracy (test normals)": f"{matrix.loc[name, name] / matrix.loc[name].sum():.2f}",
                    **{METRICS[m]: f"{section[m]:.2f}" for m in METRICS},
                }
            )
    table = pd.DataFrame(rows).sort_values(["section", "validation_split"])
    out += [markdown(table), ""]
    summary_path = root / "dcase2022" / "summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path, dtype={"section": str})
        summary = summary[summary["machine_type"] == "ToyCar"]
        spread = summary.assign(
            value=[
                f"{m:.2f} ± {s:.2f}"
                for m, s in zip(summary["mean"], summary["std"], strict=True)
            ],
            metric=summary["metric"].map(METRICS),
        ).pivot(index="section", columns="metric", values="value")
        out += [
            "Fidelity runs on ToyCar (validation_split 0.1, 5 seeds, mean ± std):",
            "",
            markdown(spread.reset_index()),
            "",
        ]
    return out


def controls(root: Path) -> list[str]:
    """ID accuracy, confusion matrices, curves, AUC and determinism."""
    out = ["## Internal controls (study data, main split)", ""]
    if not root.exists():
        return [*out, "Not run.", ""]
    result = json.loads((root / "controls.json").read_text())
    out += [
        f"Training condition {result['condition']}, seed {result['seed']}, commit "
        f"`{result['git_commit']}`. Determinism ({result['determinism_group']} "
        f"retrained with the same seed, identical scores): "
        f"{'yes' if result['deterministic'] else 'NO'}.",
        "",
    ]
    accuracy_rows, auc_rows = [], []
    for group, record in result["groups"].items():
        history = record["history"]
        first, last = history[0], history[-1]
        accuracy_row = {
            "machine_type": group,
            "train loss (first → last epoch)": f"{first['loss']:.3f} → {last['loss']:.3f}",
            "train accuracy": (
                f"{first['accuracy']:.3f} → {last['accuracy']:.3f}"
                if "accuracy" in first
                else "—"
            ),
        }
        for snr, per_snr in record["per_snr"].items():
            accuracy_row[f"ID accuracy, test {int(snr):+d} dB"] = (
                f"{per_snr['id_accuracy']:.3f}"
            )
            for s in per_snr["scores"]:
                auc_rows.append(
                    {
                        "machine_type": s.get("machine_type", group),
                        "machine_id": s["machine_id"],
                        "test_snr": int(snr),
                        "auc": s["auc"],
                    }
                )
        accuracy_rows.append(accuracy_row)
    out += [
        "### Training and ID accuracy",
        "",
        markdown(pd.DataFrame(accuracy_rows)),
        "",
    ]
    out += [
        "### Confusion matrices (normal test clips; rows true, columns predicted)",
        "",
    ]
    for group, record in result["groups"].items():
        for snr, per_snr in record["per_snr"].items():
            matrix = pd.DataFrame(per_snr["confusion"]).T
            matrix.index.name = "true"
            out += [
                f"**{group}, test {int(snr):+d} dB**",
                "",
                markdown(matrix.reset_index()),
                "",
            ]
    table = pd.DataFrame(auc_rows).pivot_table(
        index=["machine_type", "machine_id"], columns="test_snr", values="auc"
    )
    table.columns = [f"test {snr:+d} dB" for snr in table.columns]
    out += [
        "### AUC per machine ID (one seed; implementation check, not a result)",
        "",
        markdown(table.map("{:.3f}".format).reset_index()),
        "",
    ]
    return out


def main(argv: list[str] | None = None) -> int:
    """Prints the Markdown tables.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        0.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--condition", default="0dB")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    root = load_paths().output_root / "verification" / "mobilenetv2"
    lines = [
        "**Code validation only. The MIMII AUC values of the internal controls "
        "(training at 0 dB, one seed) are implementation checks; they are not "
        "results of the study, not even partial ones.**",
        "",
    ]
    lines += fidelity(root / "dcase2022")
    lines += validation_split_control(root)
    lines += controls(root / "internal" / args.condition / f"seed{args.seed}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
