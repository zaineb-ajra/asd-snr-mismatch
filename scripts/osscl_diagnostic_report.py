"""Tables of the OS-SCL diagnostic (not part of the study).

Usage::

    python scripts/osscl_diagnostic_report.py > results/verification/osscl_diagnostic.md

Reads ``<output_root>/verification/osscl/diagnostic/diagnostic.json`` written by
``python -m noisemismatch.diagnose_osscl``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from noisemismatch.config import load_paths

WEAK = ("valve/id_02", "valve/id_06", "fan/id_04")


def markdown(table: pd.DataFrame) -> str:
    """Renders a DataFrame as a GitHub-flavored Markdown table."""
    header = "| " + " | ".join(str(c) for c in table.columns) + " |"
    rule = "|" + "|".join("---" for _ in table.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.values]
    return "\n".join([header, rule, *rows])


def main() -> int:
    """Prints the Markdown tables."""
    root = load_paths().output_root / "verification" / "osscl" / "diagnostic"
    d = json.loads((root / "diagnostic.json").read_text())
    lines = [
        "**Diagnostic of the internal-control OS-SCL model (training at 0 dB, seed 0, "
        f"commit `{d['git_commit'][:7]}`). Not a result of the study.**",
        "",
        "### Machine-ID accuracy on the training clips, trained (raw) and EMA weights",
        "",
    ]
    rows = []
    for record in d["history"]:
        if "class_accuracy_raw" not in record:
            continue
        row = {"epoch": record["epoch"], "loss": f"{record['loss']:.2f}"}
        for name in ("raw", "ema"):
            acc = record[f"class_accuracy_{name}"]
            row[f"{name}: mean over classes"] = f"{np.mean(list(acc.values())):.3f}"
            row[f"{name}: min (class)"] = (
                f"{min(acc.values()):.3f} ({min(acc, key=acc.get)})"
            )
            for c in WEAK:
                row[f"{name}: {c}"] = f"{acc[c]:.3f}"
        rows.append(row)
    lines += [markdown(pd.DataFrame(rows)), ""]
    lines += ["### Machine-ID accuracy on the normal test clips at the end", ""]
    rows = []
    for name, per_snr in d["test_normal_accuracy"].items():
        for snr, values in per_snr.items():
            row = {
                "weights": name,
                "test SNR": f"{int(snr):+d} dB",
                "overall": f"{values['overall']:.3f}",
            }
            row.update({c: f"{values['per_class'][c]:.3f}" for c in WEAK})
            row["min (class)"] = (
                f"{min(values['per_class'].values()):.3f} "
                f"({min(values['per_class'], key=values['per_class'].get)})"
            )
            rows.append(row)
    lines += [markdown(pd.DataFrame(rows)), ""]
    for name, records in d["geometry"].items():
        table = pd.DataFrame(records)
        for column in table.columns:
            if table[column].dtype == float:
                table[column] = table[column].map("{:.3f}".format)
        table = table.rename(columns=lambda c: c.replace("_", " "))
        lines += [
            f"### Embedding geometry per class, {name} weights (training SNR)",
            "",
            markdown(table),
            "",
        ]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
