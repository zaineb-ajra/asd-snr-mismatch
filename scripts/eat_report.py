"""Tables of the EAT + kNN code validation (not part of the study).

Usage::

    python scripts/eat_report.py > results/verification/eat_knn_tables.md

Reads ``<output_root>/verification/eat_knn/{fidelity,internal/0dB}/`` written by
``python -m noisemismatch.verification_eat``.
"""

from __future__ import annotations

import json

import pandas as pd

from noisemismatch.config import load_paths


def markdown(table: pd.DataFrame) -> str:
    """Renders a DataFrame as a GitHub-flavored Markdown table."""
    header = "| " + " | ".join(str(c) for c in table.columns) + " |"
    rule = "|" + "|".join("---" for _ in table.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.values]
    return "\n".join([header, rule, *rows])


def main() -> int:
    """Prints the Markdown tables."""
    root = load_paths().output_root / "verification" / "eat_knn"
    fid = json.loads((root / "fidelity" / "summary.json").read_text())
    lines = [
        "**Code validation only. The AUC values below (reference set at 0 dB, "
        "including the sensitivity analysis) are implementation checks; they are "
        "not results of the study, not even partial ones, and must not be cited "
        "as such.**",
        "",
        "## (a) Fidelity of the encoder port",
        "",
        f"{fid['n_pass']} of {fid['n_clips']} clips pass (criterion: relative error "
        f"<= {fid['tolerance']['relative']:g}, cosine >= "
        f"{fid['tolerance']['min_cosine']}); commit `{fid['git_commit'][:7]}`; "
        f"reference: transformers {fid['reference']['transformers']}, torch "
        f"{fid['reference']['torch']}.",
        "",
        "| Array | max relative error | min cosine |",
        "|---|---|---|",
    ]
    for name in ("fbank", "time_mean", "cls"):
        cos = fid["min_cosine"].get(name)
        cos_text = "—" if cos is None else f"{cos:.12f}"
        lines.append(f"| {name} | {fid['max_rel_error'][name]:.2e} | {cos_text} |")
    ctl_root = root / "internal" / "0dB"
    ctl = json.loads((ctl_root / "summary.json").read_text())
    results = pd.read_json(ctl_root / "controls.json")
    lines += [
        "",
        "## (c) Internal controls (main split, reference set at 0 dB)",
        "",
        f"Determinism ({ctl['determinism_array']}, recomputed with a fresh "
        f"extractor, bitwise identical): {'yes' if ctl['deterministic'] else 'NO'}. "
        f"Largest self-distance of a training clip: {ctl['self_distance_max']:.1e}.",
        "",
        "### Main configuration (temporal mean, K = 1): AUC per machine ID",
        "",
    ]
    main = results[results["main"]]
    table = main.pivot_table(
        index=["machine_type", "machine_id"], columns="test_snr_db", values="auc"
    )
    table.columns = [f"test {c:+d} dB" for c in table.columns]
    lines += [markdown(table.map("{:.3f}".format).reset_index()), ""]
    lines += [
        "### Sensitivity analysis (secondary; never replaces the main configuration)",
        "",
        "Mean AUC over the 15 machine IDs (and difference with the main "
        "configuration), per test SNR:",
        "",
    ]
    means = results.groupby(["pooling", "k", "test_snr_db"])["auc"].mean().unstack()
    ref = means.loc[("time_mean", 1)]
    rows = []
    for (pooling, k), row in means.iterrows():
        label = f"{pooling}, K = {k}" + (
            " (main)" if (pooling, k) == ("time_mean", 1) else ""
        )
        rows.append(
            [label]
            + [
                f"{row[s]:.3f}"
                + (
                    ""
                    if (pooling, k) == ("time_mean", 1)
                    else f" ({row[s] - ref[s]:+.3f})"
                )
                for s in means.columns
            ]
        )
    lines += [
        markdown(
            pd.DataFrame(
                rows, columns=["variant"] + [f"test {s:+d} dB" for s in means.columns]
            )
        ),
        "",
    ]
    per_id = results.pivot_table(
        index=["machine_type", "machine_id", "test_snr_db"],
        columns=["pooling", "k"],
        values="auc",
    )
    spread = (per_id.max(axis=1) - per_id.min(axis=1)).describe()
    lines += [
        f"Spread of the AUC across the four variants, per (machine ID, test SNR), "
        f"{int(spread['count'])} cells: median {spread['50%']:.3f}, maximum "
        f"{spread['max']:.3f}.",
        "",
    ]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
