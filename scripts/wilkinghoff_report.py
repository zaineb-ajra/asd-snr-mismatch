"""Tables of the Wilkinghoff fidelity check (not part of the study).

Usage::

    python scripts/wilkinghoff_report.py > results/verification/wilkinghoff_tables.md

The pre-registered criterion (12 cells, |difference| <= 3.0 points) is evaluated
under both per-domain definitions (official DCASE: normal clips of the domain and
all anomalous clips; restricted: clips of the domain only) and both verdicts are
reported; the mixed cells do not depend on the definition.
"""

from __future__ import annotations

import pandas as pd

from noisemismatch.config import load_paths
from noisemismatch.verification_wilkinghoff import PUBLISHED, TOLERANCE

CELLS = (("source", "auc"), ("target", "auc"), ("mixed", "auc"), ("mixed", "pauc"))


def markdown(table: pd.DataFrame) -> str:
    """Renders a DataFrame as a GitHub-flavored Markdown table."""
    header = "| " + " | ".join(str(c) for c in table.columns) + " |"
    rule = "|" + "|".join("---" for _ in table.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.values]
    return "\n".join([header, rule, *rows])


def column(domain: str, metric: str, definition: str) -> str:
    """Our metric column for a published cell under a definition."""
    if domain == "mixed":
        return f"{metric}_mixed"
    return f"{metric}_{domain}_{definition}"


def main() -> int:
    """Prints the Markdown tables."""
    root = load_paths().output_root / "verification" / "wilkinghoff" / "fidelity"
    published = pd.read_csv(PUBLISHED)
    lines = [
        "**Code validation only (not a result of the study).** Published values: "
        "Wilkinghoff, ICASSP 2024, Table 1, proposed approach (mean ± std, 5 trials). "
        "Ours: mean ± std over 5 seeds. Criterion fixed before the runs: 12 cells, "
        f"agreement if |difference| ≤ {TOLERANCE:.1f} points; evaluated under both "
        "per-domain definitions.",
        "",
    ]
    verdicts = {"official": [], "restricted": []}
    info_rows = []
    runs = (("2022", "", True), ("2023", "", True), ("2023", "_crop_per_step", False))
    for dataset, variant, in_verdict in runs:
        summary_path = root / f"dcase{dataset}{variant}" / "summary.csv"
        if not summary_path.exists():
            continue
        summary = pd.read_csv(summary_path)
        for part in ("dev", "eval"):
            name = f"dcase{dataset}_{part}"
            title = " — control: random crop at every training step" if variant else ""
            ours = summary[summary["set"] == part].set_index("metric")
            if ours.empty:
                continue
            rows = []
            for domain, metric in CELLS:
                cell = published[
                    (published["set"] == name)
                    & (published["domain"] == domain)
                    & (published["metric"] == metric)
                ].iloc[0]
                row = {
                    "cell": f"{metric.upper()} {domain}",
                    "published": f"{cell['mean']:.1f} ± {cell['std']:.1f}",
                }
                for definition in ("official", "restricted"):
                    values = ours.loc[column(domain, metric, definition)]
                    diff = values["mean"] - cell["mean"]
                    agrees = abs(diff) <= TOLERANCE
                    if in_verdict:
                        verdicts[definition].append(agrees)
                    row[f"ours ({definition})"] = (
                        f"{values['mean']:.1f} ± {values['std']:.1f}"
                    )
                    row[f"diff ({definition})"] = f"{diff:+.1f}"
                    row[f"agrees ({definition})"] = "yes" if agrees else "no"
                rows.append(row)
            lines += [
                f"### {name}{title} (n = {int(ours['n'].iloc[0])} seeds)",
                "",
                markdown(pd.DataFrame(rows)),
                "",
            ]
            for domain in ("source", "target"):
                cell = published[
                    (published["set"] == name)
                    & (published["domain"] == domain)
                    & (published["metric"] == "pauc")
                ].iloc[0]
                row = {
                    "set": name + variant,
                    "cell": f"pAUC {domain}",
                    "published": f"{cell['mean']:.1f} ± {cell['std']:.1f}",
                }
                for definition in ("official", "restricted"):
                    values = ours.loc[column(domain, "pauc", definition)]
                    row[f"ours ({definition})"] = (
                        f"{values['mean']:.1f} ± {values['std']:.1f}"
                    )
                    row[f"diff ({definition})"] = (
                        f"{values['mean'] - cell['mean']:+.1f}"
                    )
                info_rows.append(row)
    lines += [
        "### Verdict of the pre-registered criterion",
        "",
        f"- Official definition: {sum(verdicts['official'])} of {len(verdicts['official'])} cells agree.",
        f"- Restricted definition: {sum(verdicts['restricted'])} of {len(verdicts['restricted'])} cells agree.",
        "",
        "### For information (outside the criterion): per-domain pAUC",
        "",
        markdown(pd.DataFrame(info_rows)) if info_rows else "none",
        "",
    ]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
