# Results

Only final aggregates are versioned here:

- `aggregated/`: per-ID metrics with the 95 % bootstrap intervals of the AUC
  (`per_id_*.csv`), aggregated values per run (`aggregated.csv`), summaries over
  seeds (`summary_*.csv`), seed-paired ΔAUC (`delta_*.csv`), paired-bootstrap
  95 % intervals of the aggregated values (`bootstrap_intervals_*.csv`), ratios
  of errors (`error_ratio_main_auc.csv`) and the EAT scoring variants
  (`eat_variants_main.csv`);
- `paper/`: Table 1 (`tab_auc.tex`) and Figure 1 (`fig_pertes.pdf`) of the EGC
  2027 paper, generated from `aggregated/` by `scripts/paper_figures.py`;
- `study_report.md`: every table of the analysis, generated from `aggregated/`;
- `tables/`, `figures/`: table and figure of the study report (English);
- `phase6/`: commit of the study runs and the checklist reports of both stages
  (the commit identifiers refer to the development history, which is not part
  of this repository);
- `verification/`: Phase 4 validation of the model implementations (not study
  results).

Per-run outputs (result files with full configuration, seed, git commit and
duration; per-clip anomaly scores; logs) are written under `output_root`
(see `configs/paths.yaml`) and are not versioned.
