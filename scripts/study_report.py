"""Phase 7, step 2: tables, figure and report of the study.

Usage (after ``scripts/study_analysis.py per-id`` and ``eat-variants``)::

    python scripts/study_report.py

Reads ``results/aggregated/per_id_*.csv`` and ``eat_variants_main.csv``; writes
``results/aggregated/{aggregated,summary,delta,error_ratio}_*.csv``,
``results/study_report.md``, the LaTeX table ``results/tables/main_auc.tex`` and
the figure ``results/figures/delta_auc_vs_snr_gap.pdf``. Nothing is rerun.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from itertools import pairwise
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from noisemismatch.config import CONFIG_DIR, REPO_ROOT, load_yaml
from noisemismatch.eval.aggregate import aggregate, seed_summary
from noisemismatch.eval.study import (
    CELL_KEYS,
    CONDITION_SNR,
    MODEL_NAMES,
    MODELS,
    RUN_KEYS,
    aggregated,
    deltas,
    error_ratios,
    over_seeds,
)

matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

AGG = REPO_ROOT / "results" / "aggregated"
TABLES = REPO_ROOT / "results" / "tables"
FIGURES = REPO_ROOT / "results" / "figures"
REPORT = REPO_ROOT / "results" / "study_report.md"
PUBLISHED = REPO_ROOT / "metadata" / "published" / "huang2025_table2.csv"
CONDITIONS: list[str] = list(CONDITION_SNR)
COND_LABEL: dict[str, str] = {
    "m6dB": "−6 dB",
    "0dB": "0 dB",
    "p6dB": "+6 dB",
    "multi": "multi",
}
SNR_LABEL: dict[int, str] = {-6: "−6 dB", 0: "0 dB", 6: "+6 dB"}
TYPES: list[str] = ["fan", "pump", "slider", "valve"]
DETERMINISTIC: str = "eat_knn"
# Extreme mismatches: towards the clean side (trained at -6 dB, tested at +6 dB)
# and towards the noisy side (trained at +6 dB, tested at -6 dB).
CLEAN: tuple[str, int] = ("m6dB", 6)
NOISY: tuple[str, int] = ("p6dB", -6)
SELECTED_IDS: list[tuple[str, str, str]] = [
    ("valve", "id_02", "below chance for both ID classifiers in Phase 4"),
    ("slider", "id_00", "178 training clips (identity exclusions)"),
    ("fan", "id_00", "robustness aggregation of the protocol"),
]
# Categorical palette (dataviz reference, slots 1-5, validated for adjacent
# pairs; markers give a second encoding for print and colour vision deficiency).
COLORS: dict[str, str] = {
    "ae": "#2a78d6",
    "mobilenetv2": "#eb6834",
    "eat_knn": "#1baf7a",
    "wilkinghoff": "#eda100",
    "osscl": "#e87ba4",
}
MARKERS: dict[str, str] = {
    "ae": "o",
    "mobilenetv2": "s",
    "eat_knn": "^",
    "wilkinghoff": "D",
    "osscl": "v",
}


# ----------------------------------------------------------------- helpers


def read(name: str) -> pd.DataFrame:
    """Reads a CSV of ``results/aggregated``."""
    return pd.read_csv(AGG / name, dtype={"machine_id": str})


def both(per_id: pd.DataFrame, drop: Iterable[dict] = ()) -> pd.DataFrame:
    """Aggregated AUC and pAUC per (run, test SNR)."""
    drop = list(drop)
    return aggregated(per_id, "auc", drop).merge(
        aggregated(per_id, "pauc", drop), on=[*RUN_KEYS, "test_snr_db"]
    )


def fmt(
    mean: float, std: float, model: str, scale: float = 100.0, digits: int = 1
) -> str:
    """``mean ± std`` (EAT: mean only, its three seeds are identical)."""
    if model == DETERMINISTIC:
        return f"{mean * scale:.{digits}f}"
    return f"{mean * scale:.{digits}f} ± {std * scale:.{digits}f}"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    """A Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def lookup(summary: pd.DataFrame) -> dict[tuple, pd.Series]:
    """Summary rows by (model, training condition, test SNR)."""
    return {tuple(row[k] for k in CELL_KEYS): row for _, row in summary.iterrows()}


def matrix(
    cells: dict[tuple, pd.Series],
    snrs: list[int],
    conditions: list[str],
    render: Callable[[str, pd.Series], str],
    diagonal: str | None = None,
) -> str:
    """Model x training condition rows, test SNR columns."""
    rows = []
    for model in MODELS:
        for condition in conditions:
            row = [
                MODEL_NAMES[model] if condition == conditions[0] else "",
                COND_LABEL[condition],
            ]
            for snr in snrs:
                if diagonal is not None and CONDITION_SNR[condition] == snr:
                    row.append(diagonal)
                else:
                    row.append(render(model, cells[(model, condition, snr)]))
            rows.append(row)
    return md_table(
        ["Model", "Training", *[f"Test {SNR_LABEL[s]}" for s in snrs]], rows
    )


def per_type(per_id: pd.DataFrame) -> pd.DataFrame:
    """Per-type values with ``model`` = ``<model>/<type>`` (for :func:`deltas`)."""
    table = (
        per_id.groupby([*RUN_KEYS, "test_snr_db", "machine_type"])[["auc", "pauc"]]
        .mean()
        .reset_index()
    )
    table["model"] = table["model"] + "/" + table["machine_type"]
    return table.drop(columns="machine_type")


# ----------------------------------------------------------------- sections


def section_main(
    summary: dict[str, pd.DataFrame],
    delta: dict[str, pd.DataFrame],
    ratio: pd.DataFrame,
) -> list[str]:
    """Main matrices: AUC, pAUC, ΔAUC, ΔpAUC, error ratio."""
    snrs = [-6, 0, 6]
    auc_cells = lookup(summary["auc"])
    pauc_cells = lookup(summary["pauc"])
    d_cells = lookup(delta["auc"])
    dp_cells = lookup(delta["pauc"])
    r_cells = lookup(ratio)
    out = [
        "## 1. Main split: AUC and pAUC (%, mean ± standard deviation over 3 seeds)",
        "",
        "Aggregation of the protocol (mean over machine IDs, then over machine types). "
        "EAT + kNN involves no random choice: its three seeds are identical and only "
        "the value is given.",
        "",
        "### AUC",
        "",
        matrix(auc_cells, snrs, CONDITIONS, lambda m, r: fmt(r["mean"], r["std"], m)),
        "",
        "### pAUC (max FPR 0.1, McClish-standardized)",
        "",
        matrix(pauc_cells, snrs, CONDITIONS, lambda m, r: fmt(r["mean"], r["std"], m)),
        "",
        "## 2. ΔAUC (points), computed seed by seed against the matched model",
        "",
        "ΔAUC(s → s′) = AUC(train s, test s′) − AUC(train s′, test s′); multi: "
        "train on all three SNRs. Mean ± standard deviation over seeds of the "
        "per-seed differences (pre-registered primary measure).",
        "",
        matrix(
            d_cells, snrs, CONDITIONS, lambda m, r: fmt(r["mean"], r["std"], m), "—"
        ),
        "",
        "### ΔpAUC",
        "",
        matrix(
            dp_cells, snrs, CONDITIONS, lambda m, r: fmt(r["mean"], r["std"], m), "—"
        ),
        "",
        "## 3. Secondary analysis (decided after the stage-A results): ratio of errors",
        "",
        "(1 − AUC(train s, test s′)) / (1 − AUC(train s′, test s′)), on the aggregated "
        "AUCs (a ratio of aggregates, not a mean of per-ID ratios), seed by seed, then "
        "mean ± standard deviation over seeds. Unstable when the matched AUC is close "
        'to 1 (protocol, "Analyses added after the results").',
        "",
        matrix(
            r_cells,
            snrs,
            CONDITIONS,
            lambda m, r: fmt(r["mean"], r["std"], m, scale=1.0, digits=2),
            "—",
        ),
        "",
    ]
    return out


def section_fragility(delta: pd.DataFrame, ratio: pd.DataFrame) -> list[str]:
    """Most fragile model per mismatched cell under the two measures."""
    d_cells, r_cells = lookup(delta), lookup(ratio)
    rows = []
    for condition in ("m6dB", "0dB", "p6dB"):
        for snr in (-6, 0, 6):
            if CONDITION_SNR[condition] == snr:
                continue
            by_delta = min(MODELS, key=lambda m: d_cells[(m, condition, snr)]["mean"])
            by_ratio = max(MODELS, key=lambda m: r_cells[(m, condition, snr)]["mean"])
            d, r = (
                d_cells[(by_delta, condition, snr)],
                r_cells[(by_ratio, condition, snr)],
            )
            rows.append(
                [
                    f"{COND_LABEL[condition]} → {SNR_LABEL[snr]}",
                    MODEL_NAMES[by_delta],
                    fmt(d["mean"], d["std"], by_delta),
                    MODEL_NAMES[by_ratio],
                    fmt(r["mean"], r["std"], by_ratio, scale=1.0, digits=2),
                    "same" if by_delta == by_ratio else "differ",
                ]
            )
    detail = []
    for model in MODELS:
        d, r = d_cells[(model, *NOISY)], r_cells[(model, *NOISY)]
        detail.append(
            [
                MODEL_NAMES[model],
                fmt(d["matched"], 0.0, DETERMINISTIC),
                fmt(d["mean"], d["std"], model),
                fmt(r["mean"], r["std"], model, scale=1.0, digits=2),
            ]
        )
    return [
        "## 4. Most fragile model per mismatched cell: ΔAUC in points vs ratio of errors",
        "",
        md_table(
            [
                "Training → test",
                "Largest loss in points",
                "ΔAUC",
                "Largest error ratio",
                "Ratio",
                "Designated model",
            ],
            rows,
        ),
        "",
        "Trained at +6 dB, tested at −6 dB (matched AUC: model trained and tested at "
        "−6 dB, mean over seeds):",
        "",
        md_table(["Model", "Matched AUC", "ΔAUC", "Error ratio"], detail),
        "",
    ]


def section_family(delta_runs: pd.DataFrame, per_id_delta: pd.DataFrame) -> list[str]:
    """Both directions of the mismatch for every model, per seed and per ID."""
    rows = []
    for model in MODELS:
        part = delta_runs[delta_runs["model"] == model]

        def cell(condition: str, snr: int, part: pd.DataFrame = part) -> np.ndarray:
            sel = part[
                (part["train_condition"] == condition) & (part["test_snr_db"] == snr)
            ]
            return sel.sort_values("seed")["delta"].to_numpy()

        clean, noisy = cell(*CLEAN), cell(*NOISY)
        dominant = [
            "clean" if c < n else "noisy" for c, n in zip(clean, noisy, strict=True)
        ]
        ids = per_id_delta[per_id_delta["model"] == model]
        n_clean = int((ids["clean"] < ids["noisy"]).sum())
        rows.append(
            [
                MODEL_NAMES[model],
                fmt(*seed_summary(clean), model),
                " / ".join(f"{100 * v:.1f}" for v in clean),
                fmt(*seed_summary(noisy), model),
                " / ".join(f"{100 * v:.1f}" for v in noisy),
                " / ".join(dominant),
                f"{n_clean} of {len(ids)}",
            ]
        )
    six = []
    for model in MODELS:
        part = delta_runs[delta_runs["model"] == model]
        values = []
        for condition, snr in (("m6dB", 0), ("0dB", 6), ("p6dB", 0), ("0dB", -6)):
            sel = part[
                (part["train_condition"] == condition) & (part["test_snr_db"] == snr)
            ]
            values.append(fmt(*seed_summary(sel.sort_values("seed")["delta"]), model))
        six.append([MODEL_NAMES[model], *values])
    return [
        "## 5. Direction of the larger loss (the family dichotomy is abandoned, see paper notes)",
        "",
        "ΔAUC towards the clean side: trained at −6 dB, tested at +6 dB; towards the "
        "noisy side: trained at +6 dB, tested at −6 dB. Per seed (seeds 0 / 1 / 2) and "
        "mean ± standard deviation. Last column: machine IDs (of 15) whose loss is "
        "larger towards the clean side (per-ID ΔAUC, mean over seeds).",
        "",
        md_table(
            [
                "Model",
                "ΔAUC clean side",
                "per seed",
                "ΔAUC noisy side",
                "per seed",
                "larger loss, per seed",
                "IDs with larger loss on the clean side",
            ],
            rows,
        ),
        "",
        "Mismatches of 6 dB (ΔAUC, mean ± standard deviation over seeds):",
        "",
        md_table(
            [
                "Model",
                "−6 → 0 dB (clean side)",
                "0 → +6 dB (clean side)",
                "+6 → 0 dB (noisy side)",
                "0 → −6 dB (noisy side)",
            ],
            six,
        ),
        "",
    ]


def section_robustness(agg: dict[str, pd.DataFrame]) -> list[str]:
    """ΔAUC under the strict variant and without fan id_00, against the main one."""
    main = deltas(agg["main"]).set_index([*RUN_KEYS, "test_snr_db"])["delta"]
    rows = []
    for model in MODELS:
        row = [MODEL_NAMES[model]]
        for name in ("strict", "fan_without_id_00"):
            other = deltas(agg[name]).set_index([*RUN_KEYS, "test_snr_db"])["delta"]
            diff = (other - main).xs(model, level="model").abs()
            row.append(f"{100 * diff.max():.1f}")
            noisy = deltas(agg[name])
            noisy = noisy[
                (noisy["model"] == model)
                & (noisy["train_condition"] == NOISY[0])
                & (noisy["test_snr_db"] == NOISY[1])
            ]
            row.append(fmt(*seed_summary(noisy.sort_values("seed")["delta"]), model))
        base = deltas(agg["main"])
        base = base[
            (base["model"] == model)
            & (base["train_condition"] == NOISY[0])
            & (base["test_snr_db"] == NOISY[1])
        ]
        row.insert(1, fmt(*seed_summary(base.sort_values("seed")["delta"]), model))
        rows.append(row)
    return [
        "## 6. Robustness: strict identity variant and fan without id_00 (pre-registered)",
        "",
        "Largest absolute change of a per-seed ΔAUC (any cell, any seed) against the "
        "main evaluation, in points, and the ΔAUC trained at +6 dB / tested at −6 dB "
        "under each evaluation.",
        "",
        md_table(
            [
                "Model",
                "ΔAUC +6 → −6, main",
                "strict: largest change",
                "strict: ΔAUC +6 → −6",
                "without fan id_00: largest change",
                "without fan id_00: ΔAUC +6 → −6",
            ],
            rows,
        ),
        "",
    ]


def section_sensitivity(
    summary: dict[str, pd.DataFrame], delta: dict[str, pd.DataFrame]
) -> list[str]:
    """2x2 sensitivity split against the corresponding cells of the main split."""
    rows, order = [], []
    main_cells = lookup(summary["main"])
    sens_cells = lookup(summary["sensitivity"])
    dm, ds = lookup(delta["main"]), lookup(delta["sensitivity"])
    for model in MODELS:
        for condition in ("0dB", "p6dB"):
            row = [
                MODEL_NAMES[model] if condition == "0dB" else "",
                COND_LABEL[condition],
            ]
            for snr in (0, 6):
                m, s = (
                    main_cells[(model, condition, snr)],
                    sens_cells[(model, condition, snr)],
                )
                row.append(
                    f"{fmt(m['mean'], m['std'], model)} / {fmt(s['mean'], s['std'], model)}"
                )
            snr = 6 if condition == "0dB" else 0
            m, s = dm[(model, condition, snr)], ds[(model, condition, snr)]
            row.append(
                f"{fmt(m['mean'], m['std'], model)} / {fmt(s['mean'], s['std'], model)}"
            )
            rows.append(row)
    for split in ("main", "sensitivity"):
        cells = lookup(summary[split])
        for snr, condition in ((0, "0dB"), (6, "p6dB")):
            ranked = sorted(MODELS, key=lambda m: -cells[(m, condition, snr)]["mean"])
            order.append(
                [
                    split,
                    f"matched {SNR_LABEL[snr]}",
                    " > ".join(MODEL_NAMES[m] for m in ranked),
                ]
            )
    return [
        "## 7. Sensitivity split (2×2, all clips identifiable between 0 and +6 dB)",
        "",
        "Main split / sensitivity split, AUC (%) and ΔAUC (points), mean ± standard "
        "deviation over seeds.",
        "",
        md_table(
            ["Model", "Training", "Test 0 dB", "Test +6 dB", "ΔAUC (mismatched cell)"],
            rows,
        ),
        "",
        "Order of the models in the matched cells:",
        "",
        md_table(["Split", "Cell", "Order (highest AUC first)"], order),
        "",
    ]


def section_per_machine(
    per_id_main: pd.DataFrame, type_summary: pd.DataFrame, type_delta: pd.DataFrame
) -> list[str]:
    """Per machine type, selected machine IDs, width of the intervals."""
    cells, dcells = lookup(type_summary), lookup(type_delta)
    rows = []
    for model in MODELS:
        for machine_type in TYPES:
            key = f"{model}/{machine_type}"
            row = [MODEL_NAMES[model] if machine_type == TYPES[0] else "", machine_type]
            for condition, snr in (("m6dB", -6), ("0dB", 0), ("p6dB", 6)):
                r = cells[(key, condition, snr)]
                row.append(fmt(r["mean"], r["std"], model))
            for condition, snr in (CLEAN, NOISY):
                r = dcells[(key, condition, snr)]
                row.append(fmt(r["mean"], r["std"], model))
            rows.append(row)
    selected = []
    matched = per_id_main[
        per_id_main["train_condition"].map(CONDITION_SNR) == per_id_main["test_snr_db"]
    ]
    for machine_type, machine_id, why in SELECTED_IDS:
        for model in MODELS:
            row = [
                f"{machine_type} {machine_id}" if model == MODELS[0] else "",
                MODEL_NAMES[model],
            ]
            for snr in (-6, 0, 6):
                sel = matched[
                    (matched["model"] == model)
                    & (matched["machine_type"] == machine_type)
                    & (matched["machine_id"] == machine_id)
                    & (matched["test_snr_db"] == snr)
                ]
                row.append(
                    f"{100 * sel['auc'].mean():.1f} "
                    f"[{100 * sel['auc_ci_low'].min():.1f}, {100 * sel['auc_ci_high'].max():.1f}]"
                )
            row.append(why if model == MODELS[0] else "")
            selected.append(row)
    widths = []
    for model in MODELS:
        sel = matched[matched["model"] == model]
        width = sel["auc_ci_high"] - sel["auc_ci_low"]
        chance = sel[(sel["auc_ci_low"] <= 0.5) & (sel["auc_ci_high"] >= 0.5)]
        names = sorted(
            {f"{r.machine_type} {r.machine_id}" for r in chance.itertuples()}
        )
        widths.append(
            [
                MODEL_NAMES[model],
                f"{100 * width.median():.1f}",
                f"{100 * width.max():.1f}",
                f"{len(chance)} of {len(sel)}"
                + (f" ({', '.join(names)})" if names else ""),
            ]
        )
    return [
        "## 8. Per machine",
        "",
        "### Per machine type: matched AUC and ΔAUC of the two extreme mismatches",
        "",
        md_table(
            [
                "Model",
                "Type",
                "AUC −6/−6",
                "AUC 0/0",
                "AUC +6/+6",
                "ΔAUC −6 → +6",
                "ΔAUC +6 → −6",
            ],
            rows,
        ),
        "",
        "### Selected machine IDs, matched cells",
        "",
        "Per-ID AUC (%), mean over seeds, with the envelope of the three per-seed 95 % "
        "bootstrap intervals [lowest lower bound, highest upper bound].",
        "",
        md_table(["Machine", "Model", "−6/−6", "0/0", "+6/+6", "Why"], selected),
        "",
        "### Width of the per-ID 95 % bootstrap intervals (matched cells, all seeds)",
        "",
        md_table(
            [
                "Model",
                "median width (points)",
                "largest width",
                "per-seed intervals containing 0.5 (chance)",
            ],
            widths,
        ),
        "",
    ]


def section_wilkinghoff_level(
    per_id: dict[str, pd.DataFrame], type_summary: pd.DataFrame
) -> list[str]:
    """The three checks on the level of Wilkinghoff, and the comparison 1b of OS-SCL."""
    main = per_id["main"]
    matched = main[main["train_condition"].map(CONDITION_SNR) == main["test_snr_db"]]
    by_id = matched.groupby(["model", "test_snr_db", "machine_type", "machine_id"])[
        "auc"
    ].mean()
    rows = []
    for model in MODELS:
        for snr in (-6, 0, 6):
            values = by_id.xs((model, snr), level=("model", "test_snr_db"))
            lowest = values.idxmin()
            rows.append(
                [
                    MODEL_NAMES[model] if snr == -6 else "",
                    SNR_LABEL[snr],
                    f"{100 * values.min():.1f} ({lowest[0]} {lowest[1]})",
                    f"{100 * values.median():.1f}",
                    f"{100 * values.max():.1f}",
                    f"{int((values >= 0.99).sum())}",
                    f"{int((values < 0.90).sum())}",
                ]
            )
    wilk = []
    for (machine_type, machine_id), _ in by_id.xs(
        ("wilkinghoff", -6), level=("model", "test_snr_db")
    ).items():
        wilk.append(
            [f"{machine_type} {machine_id}"]
            + [
                f"{100 * by_id[('wilkinghoff', snr, machine_type, machine_id)]:.1f}"
                for snr in (-6, 0, 6)
            ]
        )
    published = pd.read_csv(PUBLISHED, comment="#")
    cells = lookup(type_summary)
    compare = []
    for model in ("wilkinghoff", "osscl"):
        values = [cells[(f"{model}/{t}", "m6dB", -6)]["mean"] for t in TYPES]
        compare.append(
            [f"{MODEL_NAMES[model]} (ours, MIMII −6 dB, 15 classes)"]
            + [f"{100 * v:.1f}" for v in values]
            + [f"{100 * np.mean(values):.1f}"]
        )
    for system in ("TFSTgram(OS-SCL)", "CLP-SCF", "STgram-MFN"):
        part = published[published["system"] == system].set_index("machine_type")["auc"]
        values = [float(part[t]) for t in TYPES]
        compare.append(
            [f"{system} (published, DCASE 2020 dev)"]
            + [f"{v:.2f}" for v in values]
            + [f"{np.mean(values):.2f}"]
        )
    return [
        "## 9. Level of Wilkinghoff (checks decided after the stage-A results) and OS-SCL",
        "",
        "### a. Per machine ID, matched cells (per-ID AUC, mean over seeds)",
        "",
        md_table(
            [
                "Model",
                "Matched SNR",
                "lowest ID",
                "median",
                "highest",
                "IDs ≥ 99",
                "IDs < 90",
            ],
            rows,
        ),
        "",
        "Wilkinghoff, every machine ID (AUC %, matched cells):",
        "",
        md_table(["Machine", "−6/−6", "0/0", "+6/+6"], wilk),
        "",
        "### b. Trained and tested at −6 dB against published DCASE 2020 values",
        "",
        "Published values: Huang et al. 2025, Table II (`metadata/published/"
        "huang2025_table2.csv`); DCASE 2020 Task 2 development set, whose MIMII part "
        "(fan, pump, slider, valve) corresponds to channel 0 of the MIMII recordings "
        "at −6 dB. Not the same test split, nor the same number of classes "
        "(41 machine IDs in DCASE 2020 development and additional training sets, 15 "
        "here). Wilkinghoff's SSL model has no published DCASE 2020 result (its paper "
        "reports DCASE 2022 and 2023).",
        "",
        md_table(["System", *TYPES, "mean of the 4 types"], compare),
        "",
        "### c. Order of the models: see section 7 (sensitivity split).",
        "",
    ]


def section_eat_variants(variants: pd.DataFrame) -> list[str]:
    """Pre-registered sensitivity analysis of the EAT scoring."""
    out = []
    for (pooling, k), part in variants.groupby(["pooling", "k"], sort=False):
        values = {}
        for (condition, snr), cell in part.groupby(["train_condition", "test_snr_db"]):
            values[(condition, snr)] = aggregate(cell, "auc")[1]
        row = [
            f"{pooling}, K = {k}" + (" (main)" if bool(part["main"].iloc[0]) else "")
        ]
        for condition in CONDITIONS:
            row.append(
                " / ".join(
                    f"{100 * values[(condition, snr)]:.1f}" for snr in (-6, 0, 6)
                )
            )
        row.append(f"{100 * (values[CLEAN] - values[('p6dB', 6)]):.1f}")
        row.append(f"{100 * (values[NOISY] - values[('m6dB', -6)]):.1f}")
        out.append(row)
    return [
        "## 10. EAT + kNN scoring variants (pre-registered sensitivity analysis)",
        "",
        "AUC (%) at test −6 / 0 / +6 dB for each training condition, and the two "
        "extreme ΔAUC. The main configuration is fixed; the variants show how much the "
        "level depends on pooling and K, they are not a selection.",
        "",
        md_table(
            [
                "Pooling, K",
                *[f"train {COND_LABEL[c]}" for c in CONDITIONS],
                "ΔAUC −6 → +6",
                "ΔAUC +6 → −6",
            ],
            out,
        ),
        "",
    ]


def section_intervals(main: pd.DataFrame, sensitivity: pd.DataFrame) -> list[str]:
    """Paired-bootstrap intervals on aggregated values (decided after the results)."""

    def pts(r: pd.Series) -> str:
        return (
            f"{100 * r['estimate']:.1f} [{100 * r['low']:.1f}, {100 * r['high']:.1f}]"
        )

    def rat(r: pd.Series) -> str:
        return f"{r['estimate']:.2f} [{r['low']:.2f}, {r['high']:.2f}]"

    def get(
        table: pd.DataFrame, statistic: str, model: str, cell: tuple, other: str = ""
    ) -> pd.Series:
        sel = table[
            (table["statistic"] == statistic)
            & (table["model"] == model)
            & (table["model_b"].fillna("") == other)
            & (table["train_condition"] == cell[0])
            & (table["test_snr_db"] == str(cell[1]))
        ]
        if len(sel) != 1:
            raise KeyError((statistic, model, other, cell))
        return sel.iloc[0]

    cells = [
        (c, snr) for c in CONDITIONS for snr in (-6, 0, 6) if CONDITION_SNR[c] != snr
    ]
    delta_rows = [
        [f"{COND_LABEL[c]} → {SNR_LABEL[snr]}"]
        + [pts(get(main, "delta", m, (c, snr))) for m in MODELS]
        for c, snr in cells
    ]
    sens_rows = [
        [f"{COND_LABEL[c]} → {SNR_LABEL[snr]}"]
        + [pts(get(sensitivity, "delta", m, (c, snr))) for m in MODELS]
        for c, snr in (("0dB", 6), ("p6dB", 0))
    ]
    ratio_rows = [
        [f"{COND_LABEL[c]} → {SNR_LABEL[snr]}"]
        + [rat(get(main, "ratio", m, (c, snr))) for m in MODELS]
        for c, snr in cells
    ]
    direction = [
        [MODEL_NAMES[m], pts(get(main, "direction", m, ("m6dB-p6dB", "6/-6")))]
        for m in MODELS
    ]

    def ordered_pair(statistic: str, cell: tuple, first: str, second: str) -> str:
        if MODELS.index(first) < MODELS.index(second):
            r = get(main, statistic, first, cell, second)
            sign = 1.0
        else:
            r = get(main, statistic, second, cell, first)
            sign = -1.0
        r = r.copy()
        if sign < 0:
            r["estimate"], r["low"], r["high"] = -r["estimate"], -r["high"], -r["low"]
        return pts(r) if statistic.startswith("delta") else rat(r)

    fragile = []
    for c, snr in cells:
        if c == "multi":
            continue
        by_points = sorted(
            MODELS, key=lambda m: get(main, "delta", m, (c, snr))["estimate"]
        )
        by_ratio = sorted(
            MODELS, key=lambda m: -get(main, "ratio", m, (c, snr))["estimate"]
        )
        fragile.append(
            [
                f"{COND_LABEL[c]} → {SNR_LABEL[snr]}",
                f"{MODEL_NAMES[by_points[0]]} − {MODEL_NAMES[by_points[1]]}",
                ordered_pair("delta_difference", (c, snr), by_points[0], by_points[1]),
                f"{MODEL_NAMES[by_ratio[0]]} − {MODEL_NAMES[by_ratio[1]]}",
                ordered_pair("ratio_difference", (c, snr), by_ratio[0], by_ratio[1]),
            ]
        )
    clean_order = sorted(MODELS, key=lambda m: get(main, "delta", m, CLEAN)["estimate"])
    steps = [
        [
            f"{MODEL_NAMES[a]} − {MODEL_NAMES[b]}",
            ordered_pair("delta_difference", CLEAN, a, b),
        ]
        for a, b in pairwise(clean_order)
    ]
    header = ["Training → test", *[MODEL_NAMES[m] for m in MODELS]]
    return [
        "## 11. Intervals on aggregated values (decided on 2026-10-04, after the results)",
        "",
        "95 % percentile intervals of a paired bootstrap of the test clips: per machine "
        "ID, test identities resampled with replacement, normal and anomalous "
        "separately; 1000 resamples; the same draws for every model, training "
        "condition, seed and test SNR, so that differences between conditions and "
        "between models are paired. Point estimate: full test set. The interval "
        "reflects the sampling uncertainty of the test clips of these 15 machines, not "
        "the generalization to other machines (machine IDs are not resampled), and not "
        "the training randomness (standard deviation over seeds, sections 1–2).",
        "",
        "### ΔAUC (points, mean over seeds) [95 % interval], main split",
        "",
        md_table(header, delta_rows),
        "",
        "### ΔAUC, sensitivity split",
        "",
        md_table(header, sens_rows),
        "",
        "### Direction of the larger loss: ΔAUC(−6 → +6) − ΔAUC(+6 → −6)",
        "",
        "Negative: larger loss towards the clean side.",
        "",
        md_table(["Model", "difference [95 % interval]"], direction),
        "",
        "### Loss towards the clean side (−6 → +6): steps between consecutive models",
        "",
        md_table(
            ["Pair (larger loss first)", "difference of ΔAUC [95 % interval]"], steps
        ),
        "",
        "### Most fragile model against the runner-up, under each measure",
        "",
        md_table(
            [
                "Training → test",
                "points: pair",
                "ΔAUC difference",
                "ratio: pair",
                "ratio difference",
            ],
            fragile,
        ),
        "",
        "### Ratio of errors (secondary) [95 % interval]",
        "",
        md_table(header, ratio_rows),
        "",
    ]


# ----------------------------------------------------------------- outputs


def latex_main_table(summary: dict[str, pd.DataFrame], path: Path) -> None:
    """Main table (AUC and pAUC, main split) for IEEEtran (``table*``)."""
    auc_cells, pauc_cells = lookup(summary["auc"]), lookup(summary["pauc"])

    def tex(model: str, row: pd.Series) -> str:
        if model == DETERMINISTIC:
            return f"{100 * row['mean']:.1f}"
        return f"{100 * row['mean']:.1f}\\,$\\pm$\\,{100 * row['std']:.1f}"

    lines = [
        "% Generated by scripts/study_report.py from results/aggregated; do not edit.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{AUC and pAUC (\\%) on MIMII, mean $\\pm$ standard deviation over three "
        "seeds. Rows: training SNR; columns: test SNR. Multi: one model trained on the "
        "three SNRs with the same number of optimization steps. EAT\\,+\\,kNN involves no "
        "random choice (identical seeds).}",
        "\\label{tab:main}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        " & & \\multicolumn{3}{c}{AUC} & \\multicolumn{3}{c}{pAUC} \\\\",
        "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}",
        "Model & Train & $-6$\\,dB & 0\\,dB & $+6$\\,dB & $-6$\\,dB & 0\\,dB & $+6$\\,dB \\\\",
        "\\midrule",
    ]
    train = {
        "m6dB": "$-6$\\,dB",
        "0dB": "0\\,dB",
        "p6dB": "$+6$\\,dB",
        "multi": "multi",
    }
    for i, model in enumerate(MODELS):
        if i:
            lines.append("\\midrule")
        for condition in CONDITIONS:
            name = (
                MODEL_NAMES[model]
                .replace("+", "\\,+\\,")
                .replace(" \\,+\\, ", "\\,+\\,")
            )
            cells = [
                tex(model, auc_cells[(model, condition, s)]) for s in (-6, 0, 6)
            ] + [tex(model, pauc_cells[(model, condition, s)]) for s in (-6, 0, 6)]
            lead = name if condition == CONDITIONS[0] else ""
            lines.append(
                f"{lead} & {train[condition]} & " + " & ".join(cells) + " \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def figure_gap(delta_runs: pd.DataFrame, path: Path) -> None:
    """ΔAUC against the SNR gap (test - training), one line per model.

    Cells with the same gap (two at +-6 dB) are averaged seed by seed; error bars:
    standard deviation over seeds.
    """
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
        }
    )
    single = delta_runs[delta_runs["train_condition"] != "multi"].copy()
    single["gap"] = single["test_snr_db"] - single["train_condition"].map(CONDITION_SNR)
    # Cells with the same gap (two at ±6 dB) are averaged seed by seed.
    per_seed = single.groupby(["model", "gap", "seed"])["delta"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.axhline(0, color="#8f8e89", linewidth=0.6, zorder=1)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.5)
    ax.set_axisbelow(True)
    offsets = np.linspace(-0.8, 0.8, len(MODELS))
    for offset, model in zip(offsets, MODELS, strict=True):
        part = per_seed[per_seed["model"] == model]
        stats = (
            part.groupby("gap")["delta"]
            .agg(["mean", "std"])
            .reindex([-12, -6, 0, 6, 12])
        )
        stats.loc[0] = (0.0, 0.0)
        stats = stats.fillna(0.0)
        x = stats.index.to_numpy() + offset
        ax.errorbar(
            x,
            100 * stats["mean"],
            yerr=100 * stats["std"],
            color=COLORS[model],
            marker=MARKERS[model],
            markersize=4.5,
            markeredgecolor="white",
            markeredgewidth=0.6,
            linewidth=1.2,
            elinewidth=0.7,
            capsize=0,
            label=MODEL_NAMES[model],
            zorder=3,
        )
    ax.set_xticks([-12, -6, 0, 6, 12])
    ax.set_xticklabels(["−12", "−6", "0", "+6", "+12"])
    ax.set_xlabel("Test SNR − training SNR (dB)")
    ax.set_ylabel("ΔAUC (points)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(-12.8, 1.5, "test noisier", fontsize=7, color="#52514e", ha="left")
    ax.text(12.8, 1.5, "test cleaner", fontsize=7, color="#52514e", ha="right")
    handles = [
        Line2D(
            [],
            [],
            color=COLORS[m],
            marker=MARKERS[m],
            markersize=4.5,
            markeredgecolor="white",
            markeredgewidth=0.6,
            linewidth=1.2,
            label=MODEL_NAMES[m],
        )
        for m in MODELS
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=3,
        frameon=False,
        fontsize=7,
        handlelength=1.6,
        columnspacing=1.0,
    )
    fig.tight_layout(pad=0.3)
    fig.savefig(path, metadata={"CreationDate": None})  # reproducible file
    plt.close(fig)


# ----------------------------------------------------------------- main


def main() -> int:
    """Builds every table, the figure and the report.

    Returns:
        0 on success.
    """
    protocol = load_yaml(CONFIG_DIR / "protocol.yaml")
    drop = protocol["aggregation"]["robustness"][0]["drop"]
    per_id = {
        "main": read("per_id_main.csv"),
        "strict": read("per_id_main_strict.csv"),
        "sensitivity": read("per_id_sensitivity.csv"),
    }
    agg = {
        "main": both(per_id["main"]),
        "strict": both(per_id["strict"]),
        "fan_without_id_00": both(per_id["main"], drop),
        "sensitivity": both(per_id["sensitivity"]),
        "sensitivity_fan_without_id_00": both(per_id["sensitivity"], drop),
    }
    pd.concat([a.assign(aggregation=n) for n, a in agg.items()]).to_csv(
        AGG / "aggregated.csv", index=False, float_format="%.6f"
    )
    summary = {
        name: {v: over_seeds(a, v, CELL_KEYS) for v in ("auc", "pauc")}
        for name, a in agg.items()
    }
    delta_runs = {
        name: {v: deltas(a, v) for v in ("auc", "pauc")} for name, a in agg.items()
    }
    delta = {
        name: {
            v: over_seeds(d, "delta", CELL_KEYS).merge(
                over_seeds(d, "matched", CELL_KEYS)[[*CELL_KEYS, "mean"]].rename(
                    columns={"mean": "matched"}
                ),
                on=CELL_KEYS,
            )
            for v, d in by_value.items()
        }
        for name, by_value in delta_runs.items()
    }
    ratio_runs = error_ratios(agg["main"])
    ratio = over_seeds(ratio_runs, "ratio", CELL_KEYS)
    for name in agg:
        for v in ("auc", "pauc"):
            summary[name][v].to_csv(
                AGG / f"summary_{name}_{v}.csv", index=False, float_format="%.6f"
            )
            delta_runs[name][v].to_csv(
                AGG / f"delta_{name}_{v}.csv", index=False, float_format="%.6f"
            )
    ratio_runs.to_csv(
        AGG / "error_ratio_main_auc.csv", index=False, float_format="%.6f"
    )

    # per-ID ΔAUC of the two extreme mismatches (mean over seeds), for section 5
    main = per_id["main"]
    keyed = main.set_index([*RUN_KEYS, "test_snr_db", "machine_type", "machine_id"])[
        "auc"
    ]

    def per_id_delta(condition: str, snr: int, matched: str) -> pd.Series:
        a = keyed.xs((condition, snr), level=("train_condition", "test_snr_db"))
        b = keyed.xs((matched, snr), level=("train_condition", "test_snr_db"))
        return (a - b).groupby(level=["model", "machine_type", "machine_id"]).mean()

    family_ids = pd.DataFrame(
        {"clean": per_id_delta(*CLEAN, "p6dB"), "noisy": per_id_delta(*NOISY, "m6dB")}
    ).reset_index()

    types = per_type(main)
    type_summary = over_seeds(types, "auc", CELL_KEYS)
    type_delta = over_seeds(deltas(types), "delta", CELL_KEYS)
    type_summary.to_csv(
        AGG / "summary_main_per_type_auc.csv", index=False, float_format="%.6f"
    )

    report = [
        "# Study results (Phase 7)",
        "",
        "Generated by `scripts/study_report.py` from the per-clip scores of the Phase 6 "
        "runs (commit of the runs in `results/phase6/`). AUC and pAUC in %, ΔAUC in "
        "points. Analyses marked as decided after the results are labelled as such.",
        "",
    ]
    report += section_main(summary["main"], delta["main"], ratio)
    report += section_fragility(delta["main"]["auc"], ratio)
    report += section_family(delta_runs["main"]["auc"], family_ids)
    report += section_robustness(agg)
    report += section_sensitivity(
        {"main": summary["main"]["auc"], "sensitivity": summary["sensitivity"]["auc"]},
        {"main": delta["main"]["auc"], "sensitivity": delta["sensitivity"]["auc"]},
    )
    report += section_per_machine(main, type_summary, type_delta)
    report += section_wilkinghoff_level(per_id, type_summary)
    report += section_eat_variants(read("eat_variants_main.csv"))
    intervals = {
        split: pd.read_csv(
            AGG / f"bootstrap_intervals_{split}.csv", dtype={"test_snr_db": str}
        ).query("evaluation == 'main' and metric == 'auc'")
        for split in ("main", "sensitivity")
    }
    report += section_intervals(intervals["main"], intervals["sensitivity"])
    REPORT.write_text("\n".join(report), encoding="utf-8")
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    latex_main_table(summary["main"], TABLES / "main_auc.tex")
    figure_gap(delta_runs["main"]["auc"], FIGURES / "delta_auc_vs_snr_gap.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
