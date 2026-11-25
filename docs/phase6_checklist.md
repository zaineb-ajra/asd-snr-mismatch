# Phase 6: launch plan and checklists

Written and committed before the launch of the study (2026-10-02). The checks are
applied mechanically by `scripts/check_runs.py`; the driver
`scripts/run_study.py` starts stage B only if every check of stage A passes, and
stops otherwise. Nothing in this file is changed once runs exist
([`protocol.md`](protocol.md), "Freeze of the protocol").

## Runs

Every run is `python -m noisemismatch.run --model M --split S --train-snr C --seed K`
at the commit recorded in `<output_root>/runs/PHASE6_COMMIT`, with a clean work
tree. Per model: main split, conditions −6, 0, +6 dB and multi-condition;
sensitivity split, conditions 0 and +6 dB; seeds 0, 1, 2: 18 runs.

| Stage | Models | Runs | Processes |
|---|---|---|---|
| A | autoencoder, MobileNetV2, EAT + kNN, Wilkinghoff | 72 | two per GPU (GPUs 1 and 2) |
| B | OS-SCL | 18 | one per GPU |

The driver first checks that the feature caches of the three cached extractors
are complete (it computes any missing array before starting the processes, so
that two processes never write the same array). A run that exits with an error
is started once more from scratch after one minute (a run writes its files only
at its end, so nothing of the failed attempt remains except its log lines);
every failed attempt is listed in `<output_root>/runs/driver_failures.json` and
in the stage report. A run that fails twice is not started again: C1 then fails.

## Checklist of stage A (all must pass before stage B starts)

For each of the 72 runs:

- **C1 Completeness.** `DONE`, `result.json`, `scores.csv` and `run.log` exist.
- **C2 Provenance.** `git_commit` equals the commit of `PHASE6_COMMIT`;
  `git_dirty` is false; the recorded model configuration and protocol equal the
  committed `configs/models/<model>.yaml` and `configs/protocol.yaml`.
- **C3 Test set.** `scores.csv` has exactly one row per (test identity of the
  split, test SNR): 5,582 × 3 for the main split, 6,206 × 2 for the sensitivity
  split; no identity of the training partition appears.
- **C4 Scores.** Every score is finite.
- **C5 Metrics.** `result.json` holds one AUC and one pAUC, both in [0, 1], per
  (machine type, machine ID, test SNR): 45 for the main split, 30 for the
  sensitivity split.
- **C7 Logs.** `run.log` contains neither `Traceback` nor an `ERROR` record.

Across runs:

- **C6 Seeds.** EAT + kNN (no random choice): the three seeds give identical
  scores. Every other model: the three seeds give different scores.
- **C8 Reproduction of the validated pipelines.** MobileNetV2, main split,
  training at 0 dB, seed 0: the AUC of every (machine type, machine ID, test SNR)
  equals that of the Phase 4 internal-control run (same data, seed and code path)
  within 1e-6. EAT + kNN, main split, training at 0 dB, seed 0: the AUCs equal
  those of the Phase 4 control (`results/verification/eat_knn_controls.csv`, main
  configuration, six decimals) within 1e-6.
- **C9 Step budget.** For the autoencoder, MobileNetV2 and Wilkinghoff, on the
  main split, the number of optimization steps recorded for the multi-condition
  run equals that of the three single-SNR runs, for every training group and
  seed.
- **C10 Sanity.** For every model, split, seed and single-SNR training
  condition, the mean AUC over the machine IDs in the matched cell (test SNR =
  training SNR) is above 0.5.

## Checklist of stage B (reported; nothing is chained after it)

C1 to C5, C7, C9 and C10 as above for the 18 OS-SCL runs (C6: the seeds give
different scores), and:

- **S1 EMA safeguard.** In every run, the recorded EMA decay equals
  1 − 1 / (0.061843 × steps), and the machine-ID accuracy of the EMA weights on
  the training clips is within 0.01 of that of the trained weights for every
  class (deviation O-EMA).
- **S2 Reproduction.** Main split, training at 0 dB, seed 0: the AUC of every
  (machine type, machine ID, test SNR) equals that of the Phase 4 internal
  control of OS-SCL within 1e-6.

## Test of the check script (before the launch)

`scripts/check_runs.py` was run on synthetic run directories (random scores on
the real test tables; the EAT + kNN run at 0 dB computed for real, in memory):
without fault, both stages pass (C8 and S2 with the control AUCs copied into the
synthetic runs; the real EAT + kNN run reproduces its Phase 4 control); 28
faults introduced one at a time (missing file, other commit, dirty tree, changed
configuration or protocol, missing, duplicated, unknown or training-set row, NaN
score, missing or out-of-range metric, equal or different seeds, error in the
log, AUC shifted by 1e-5, changed or missing step count, matched AUC of 0.4, EMA
gap of 0.015, other decay, missing safeguard) are each detected by the intended
check. The run through `noisemismatch.run.execute` was also exercised for the
five models on both splits with one epoch, without writing any file.

## If a check fails

The driver writes the report, starts nothing further and stops. No run is
deleted or repeated, and no configuration is modified, before the failure has
been reported.
