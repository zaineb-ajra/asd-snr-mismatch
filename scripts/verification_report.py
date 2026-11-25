"""Prints the tables of a model's code validation (not part of the study).

Usage::

    python scripts/verification_report.py --model ae

Reads ``<output_root>/verification/<model>/summary.csv`` written by
``python -m noisemismatch.verification``.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from noisemismatch.config import load_paths


def markdown(table: pd.DataFrame) -> str:
    """Renders a DataFrame as a GitHub-flavored Markdown table."""
    header = "| " + " | ".join(str(c) for c in table.columns) + " |"
    rule = "|" + "|".join("---" for _ in table.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.values]
    return "\n".join([header, rule, *rows])


def _mean_std(values: pd.Series) -> str:
    return f"{values.mean():.3f} ± {values.std(ddof=1):.3f}"


def main() -> None:
    """Prints the per-configuration and validation-split tables."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="ae")
    args = parser.parse_args()
    summary = pd.read_csv(
        load_paths().output_root / "verification" / args.model / "summary.csv"
    )
    keys = ["pair", "split", "validation_split"]
    rows = []
    for (pair, split, fraction), part in summary.groupby(keys, sort=False):
        rows.append(
            {
                "pair": pair,
                "split": split,
                "validation_split": fraction,
                "train clips": int(part["n_train_clips"].iloc[0]),
                "test (normal/anomalous)": f"{part['n_test_normal'].iloc[0]}/{part['n_test_anomalous'].iloc[0]}",
                "AUC": _mean_std(part["auc"]),
                "pAUC": _mean_std(part["pauc"]),
                "published AUC": f"{part['published_auc'].iloc[0]:.2f}",
                "AUC - published": f"{part['auc'].mean() - part['published_auc'].iloc[0]:+.3f}",
            }
        )
    print("### AUC and pAUC (mean ± std over 3 seeds)\n")
    print(markdown(pd.DataFrame(rows)))

    rows = []
    for (pair, split), part in summary.groupby(["pair", "split"], sort=False):
        wide = part.pivot_table(index="seed", columns="validation_split", values="auc")
        diff = wide[0.1] - wide[0.0]
        spread = np.mean([wide[0.1].std(ddof=1), wide[0.0].std(ddof=1)])
        rows.append(
            {
                "pair": pair,
                "split": split,
                "AUC(0.1) - AUC(0.0), per seed": ", ".join(f"{d:+.3f}" for d in diff),
                "mean ± std": _mean_std(diff),
                "between-seed std (mean of the two)": f"{spread:.3f}",
                "|mean| / between-seed std": f"{abs(diff.mean()) / spread:.2f}",
            }
        )
    print("\n### validation_split: 90 % (Keras) minus 100 %\n")
    print(markdown(pd.DataFrame(rows)))


if __name__ == "__main__":
    main()
