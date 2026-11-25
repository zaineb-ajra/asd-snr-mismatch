"""Prints the Markdown tables of docs/data.md from the dataset metadata.

Usage::

    python scripts/make_data_tables.py   # writes docs/data_tables.md

Reads ``metadata/mimii_inventory.csv``, ``metadata/mimii_identity.csv`` and the
diagnostics under ``<data_root>/matching/`` (written by
``noisemismatch.data.prepare``, ``noisemismatch.data.matching`` and
``noisemismatch.data.identity``).
"""

from __future__ import annotations

import pandas as pd

from noisemismatch.config import CONFIG_DIR, REPO_ROOT, load_paths, load_yaml
from noisemismatch.data.identity import (
    MatchRule,
    load_identity,
    symmetry_table,
    validation_summary,
)
from noisemismatch.data.matching import MATCHING_SUBDIR
from noisemismatch.data.prepare import METADATA_DIR

GROUP = ["machine_type", "machine_id", "label"]


def markdown(table: pd.DataFrame) -> str:
    """Renders a DataFrame as a GitHub-flavored Markdown table."""
    header = "| " + " | ".join(str(c) for c in table.columns) + " |"
    rule = "|" + "|".join("---" for _ in table.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.values]
    return "\n".join([header, rule, *rows])


def counts_table(inventory: pd.DataFrame) -> pd.DataFrame:
    """Files per machine type, ID and label, checked equal at every SNR."""
    per_snr = inventory.pivot_table(index=GROUP, columns="snr_db", values="n_files")
    if not (per_snr.nunique(axis=1) == 1).all():
        raise ValueError("file counts differ across SNRs")
    counts = per_snr.iloc[:, 0].astype(int).unstack("label").reset_index()
    return counts[["machine_type", "machine_id", "normal", "abnormal"]]


def concordance_table(identity: pd.DataFrame) -> pd.DataFrame:
    """Share of accepted matches whose file has the reference's name."""
    rows = []
    for prefix, pair in (("p6", "0 ↔ +6 dB"), ("m6", "0 ↔ -6 dB")):
        for label in ("normal", "abnormal"):
            sub = identity[
                (identity["label"] == label) & identity[f"{prefix}_accepted"]
            ]
            same = sub[f"{prefix}_same_name"]
            rows.append([pair, label, len(sub), int(same.sum()), f"{same.mean():.1%}"])
    return pd.DataFrame(
        rows, columns=["SNR pair", "label", "accepted", "same name", "share"]
    )


def boundary_table(identity: pd.DataFrame) -> pd.DataFrame:
    """Accepted matches whose file lies in another machine ID or label folder."""
    rows = []
    for prefix, snr in (("p6", "+6 dB"), ("m6", "-6 dB")):
        acc = identity[identity[f"{prefix}_accepted"]]
        rows.append(
            [
                snr,
                len(acc),
                int(acc[f"{prefix}_cross_id"].sum()),
                int(acc[f"{prefix}_cross_label"].sum()),
                int((acc["label"].eq("normal") & acc[f"{prefix}_cross_label"]).sum()),
            ]
        )
    return pd.DataFrame(
        rows,
        columns=[
            "SNR",
            "accepted",
            "other ID",
            "other label",
            "normal ref. → abnormal folder",
        ],
    )


def group_validation_table(identity: pd.DataFrame) -> pd.DataFrame:
    """Per-group validation: +6 dB precision/recall against names, -6 dB consistency."""
    rows = []
    for key, d in identity.groupby(GROUP):
        acc6, same6, acc_m = d["p6_accepted"], d["p6_same_name"], d["m6_accepted"]
        rows.append(
            [
                *key,
                len(d),
                f"{(acc6 & same6).sum() / max(acc6.sum(), 1):.3f}",
                f"{(acc6 & same6).sum() / len(d):.3f}",
                f"{d.loc[acc_m, 'm6_argmax_agree'].mean():.3f}" if acc_m.any() else "-",
                f"{d['p6_score'].median():.2f}",
                f"{d['m6_score'].median():.2f}",
            ]
        )
    return pd.DataFrame(
        rows,
        columns=[
            *GROUP,
            "n",
            "+6 dB precision",
            "+6 dB recall",
            "-6 dB consistency",
            "median +6 dB score",
            "median -6 dB score",
        ],
    )


def exclusion_table(identity: pd.DataFrame) -> pd.DataFrame:
    """References, kept and excluded identities per group, with reasons."""
    reason = identity["p6_reason"].where(
        ~identity["p6_accepted"], "m6:" + identity["m6_reason"].astype(str)
    )
    reason = ("p6:" + reason).where(~identity["p6_accepted"], reason)
    table = identity.assign(reason=reason)
    counts = table.groupby(GROUP).agg(references=("clip", "size"), kept=("kept", "sum"))
    counts["excluded"] = counts["references"] - counts["kept"]
    counts["excluded %"] = (100 * counts["excluded"] / counts["references"]).round(1)
    reasons = (
        table[~table["kept"]].groupby([*GROUP, "reason"]).size().unstack(fill_value=0)
    )
    return counts.join(reasons).fillna(0).astype({"kept": int}).reset_index()


def remaining_table(identity: pd.DataFrame) -> pd.DataFrame:
    """Split sizes implied by the kept identities (MIMII protocol)."""
    kept = identity[identity["kept"]]
    per = (
        kept.groupby(["machine_type", "machine_id", "label"])
        .size()
        .unstack(fill_value=0)
    )
    total = identity.groupby(["machine_type", "machine_id", "label"]).size().unstack()
    out = pd.DataFrame(
        {
            "test anomalous": per["abnormal"],
            "test normal": per[["abnormal", "normal"]].min(axis=1),
            "train normal": per["normal"] - per[["abnormal", "normal"]].min(axis=1),
            "test anomalous (all)": total["abnormal"],
        }
    )
    return out.reset_index()


def render() -> str:
    """Builds the Markdown document with all tables."""
    matching_dir = load_paths().data_root / MATCHING_SUBDIR
    inventory = pd.read_csv(METADATA_DIR / "mimii_inventory.csv")
    identity = load_identity()
    config = load_yaml(CONFIG_DIR / "protocol.yaml")["identity"]
    rule = MatchRule(
        min_score=float(config["min_score"]),
        min_margin=float(config["min_margin"]),
        min_support=float(config["min_support_m6"]),
    )
    symmetry = symmetry_table(identity, matching_dir, rule)
    for column in ("p_binomial_half", "p_binomial_calibrated"):
        symmetry[column] = symmetry[column].map(lambda v: f"{v:.1e}")
    number = symmetry.select_dtypes("float").columns
    symmetry[number] = symmetry[number].round(3)
    screen = pd.read_csv(matching_dir / "cross_type_ch0.csv")
    screen["other type better"] = screen["best_other_type"] > screen["best_own_type"]
    cross = (
        screen.groupby(["snr_db", "machine_type"])
        .agg(
            files=("label", "size"),
            other_type_better=("other type better", "sum"),
            max_other_type=("best_other_type", "max"),
        )
        .round(3)
        .reset_index()
    )
    duplicates = pd.read_csv(matching_dir / "duplicate_pairs.csv")
    dup = (
        duplicates.groupby(["machine_type", "snr_db"])
        .size()
        .rename("pairs")
        .reset_index()
    )
    sections = [
        "<!-- Generated by scripts/make_data_tables.py; do not edit by hand. -->",
        "# MIMII: generated tables",
        "## Files per machine type, ID and label (identical at each SNR)",
        markdown(counts_table(inventory)),
        "## Validation of the identity rule (all groups)",
        markdown(pd.DataFrame([validation_summary(identity)]).round(4)),
        "## Validation per group",
        markdown(group_validation_table(identity)),
        "## Name concordance of accepted matches",
        markdown(concordance_table(identity)),
        "## Accepted matches crossing a machine ID or label boundary",
        markdown(boundary_table(identity)),
        "## Exclusions per group (reason of the first failed step)",
        markdown(exclusion_table(identity)),
        "## Kept identities per machine ID (before removing valve id_00)",
        markdown(remaining_table(identity)),
        "## Accepted matches vs non-matches (sign symmetry)",
        markdown(symmetry),
        "## Splits: sizes per machine ID (clips per SNR)",
        markdown(pd.read_csv(METADATA_DIR / "mimii_split_summary.csv")),
        "## Noise reuse check (share of test files sharing a noise segment with "
        "a training file, same-SNR score >= 0.5)",
        markdown(pd.read_csv(METADATA_DIR / "mimii_noise_check.csv").round(4)),
        "## Noise reuse across machine types (test files with a noise partner "
        "among training files of other / all machine types, same-SNR score >= 0.5)",
        markdown(
            pd.read_csv(METADATA_DIR / "mimii_noise_check_cross_type.csv").round(4)
        ),
        "## Channel-0 screen across machine types",
        markdown(cross),
        "## Pairs of distinct files of one SNR with score >= 0.5 (at +6 dB: "
        "sign-symmetric phase coincidences, see data.md)",
        markdown(dup),
    ]
    return "\n\n".join(sections) + "\n"


def main() -> None:
    """Writes docs/data_tables.md."""
    path = REPO_ROOT / "docs" / "data_tables.md"
    path.write_text(render(), encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
