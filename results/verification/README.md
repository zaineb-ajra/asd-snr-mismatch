# Code validation (not part of the study)

Results of `python -m noisemismatch.verification --model <name>`: a model trained
and tested at one SNR on a few (machine type, machine ID) pairs, with the study
split, the 2x2 sensitivity split and the original MIMII baseline split, compared
with published AUCs. These runs check the implementation only; they are never
mixed with the results of the study.

- `ae_summary.csv`: one row per (split, pair, validation_split, seed).
- `ae_tables.md`: tables printed by `scripts/verification_report.py --model ae`.
- `mobilenetv2_tables.md`: tables printed by `scripts/mobilenetv2_report.py`
  (fidelity check on DCASE 2022, `python -m noisemismatch.verification_dcase2022`;
  internal controls on the study data, `python -m noisemismatch.controls`).
- `mobilenetv2_dcase2022_summary.csv`, `mobilenetv2_dcase2022_comparison.csv`:
  mean and std over 5 seeds per (machine type, section, metric), and the
  retained cells compared with the published values ([`docs/validation.md`](../../docs/validation.md)).
- `eat_knn_tables.md`: tables printed by `scripts/eat_report.py` (fidelity
  check of the encoder port and internal controls,
  `python -m noisemismatch.verification_eat`); `eat_knn_fidelity.csv`: per-clip
  errors of the fidelity check; `eat_knn_controls.csv`: AUC, pAUC and score
  quartiles per machine ID, test SNR and scoring variant (reference set at 0 dB).
- `wilkinghoff_tables.md`: tables printed by `scripts/wilkinghoff_report.py`
  (fidelity check on DCASE 2022 and 2023, both per-domain definitions);
  `wilkinghoff_dcase*_summary.csv`, `wilkinghoff_dcase*_per_seed.csv`: harmonic
  means per set and metric, over and per seed.
- `osscl_tables.md`: internal controls of OS-SCL (`scripts/controls_report.py`),
  with the EMA decay converted to the number of steps (deviation O-EMA);
  `osscl_tables_ema0.9999.md`: the earlier control run with the repository's
  decay, kept as the evidence of the problem.
- `osscl_diagnostic.md`: diagnostic of the internal-control OS-SCL model (trained
  vs EMA weights, embedding geometry; `scripts/osscl_diagnostic_report.py`).

Per-run outputs (scores, configuration, commit) are under
`<output_root>/verification/<model>/`.
