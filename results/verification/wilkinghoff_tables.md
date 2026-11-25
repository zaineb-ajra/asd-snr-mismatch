**Code validation only (not a result of the study).** Published values: Wilkinghoff, ICASSP 2024, Table 1, proposed approach (mean ± std, 5 trials). Ours: mean ± std over 5 seeds. Criterion fixed before the runs: 12 cells, agreement if |difference| ≤ 3.0 points; evaluated under both per-domain definitions.

### dcase2022_dev (n = 5 seeds)

| cell | published | ours (official) | diff (official) | agrees (official) | ours (restricted) | diff (restricted) | agrees (restricted) |
|---|---|---|---|---|---|---|---|
| AUC source | 86.0 ± 0.9 | 83.1 ± 1.6 | -2.9 | yes | 85.3 ± 0.8 | -0.7 | yes |
| AUC target | 78.2 ± 0.7 | 75.1 ± 0.9 | -3.1 | no | 77.5 ± 1.4 | -0.7 | yes |
| AUC mixed | 82.5 ± 0.8 | 81.7 ± 1.0 | -0.8 | yes | 81.7 ± 1.0 | -0.8 | yes |
| PAUC mixed | 68.2 ± 1.1 | 68.0 ± 0.8 | -0.2 | yes | 68.0 ± 0.8 | -0.2 | yes |

### dcase2022_eval (n = 5 seeds)

| cell | published | ours (official) | diff (official) | agrees (official) | ours (restricted) | diff (restricted) | agrees (restricted) |
|---|---|---|---|---|---|---|---|
| AUC source | 77.7 ± 0.8 | 81.7 ± 1.2 | +4.0 | no | 76.4 ± 0.9 | -1.3 | yes |
| AUC target | 71.6 ± 1.0 | 58.9 ± 0.7 | -12.7 | no | 68.9 ± 0.5 | -2.7 | yes |
| AUC mixed | 74.2 ± 0.3 | 72.6 ± 0.5 | -1.6 | yes | 72.6 ± 0.5 | -1.6 | yes |
| PAUC mixed | 61.2 ± 0.3 | 60.8 ± 0.7 | -0.4 | yes | 60.8 ± 0.7 | -0.4 | yes |

### dcase2023_dev (n = 5 seeds)

| cell | published | ours (official) | diff (official) | agrees (official) | ours (restricted) | diff (restricted) | agrees (restricted) |
|---|---|---|---|---|---|---|---|
| AUC source | 71.2 ± 1.6 | 74.4 ± 1.3 | +3.2 | no | 69.1 ± 1.8 | -2.1 | yes |
| AUC target | 75.0 ± 1.5 | 63.1 ± 2.8 | -11.9 | no | 69.0 ± 2.2 | -6.0 | no |
| AUC mixed | 73.1 ± 0.9 | 69.2 ± 1.4 | -3.9 | no | 69.2 ± 1.4 | -3.9 | no |
| PAUC mixed | 57.3 ± 0.6 | 57.7 ± 0.7 | +0.4 | yes | 57.7 ± 0.7 | +0.4 | yes |

### dcase2023_dev — control: random crop at every training step (n = 5 seeds)

| cell | published | ours (official) | diff (official) | agrees (official) | ours (restricted) | diff (restricted) | agrees (restricted) |
|---|---|---|---|---|---|---|---|
| AUC source | 71.2 ± 1.6 | 73.2 ± 1.7 | +2.0 | yes | 68.7 ± 1.5 | -2.5 | yes |
| AUC target | 75.0 ± 1.5 | 64.5 ± 2.3 | -10.5 | no | 70.0 ± 2.5 | -5.0 | no |
| AUC mixed | 73.1 ± 0.9 | 69.3 ± 1.2 | -3.8 | no | 69.3 ± 1.2 | -3.8 | no |
| PAUC mixed | 57.3 ± 0.6 | 58.1 ± 0.9 | +0.8 | yes | 58.1 ± 0.9 | +0.8 | yes |

### Verdict of the pre-registered criterion

- Official definition: 6 of 12 cells agree.
- Restricted definition: 10 of 12 cells agree.

### For information (outside the criterion): per-domain pAUC

| set | cell | published | ours (official) | diff (official) | ours (restricted) | diff (restricted) |
|---|---|---|---|---|---|---|
| dcase2022_dev | pAUC source | 77.6 ± 0.8 | 77.0 ± 0.8 | -0.6 | 76.9 ± 1.1 | -0.7 |
| dcase2022_dev | pAUC target | 64.4 ± 1.1 | 66.8 ± 1.3 | +2.4 | 64.4 ± 0.9 | -0.0 |
| dcase2022_eval | pAUC source | 67.0 ± 0.5 | 69.4 ± 1.0 | +2.4 | 66.4 ± 0.9 | -0.6 |
| dcase2022_eval | pAUC target | 61.2 ± 0.9 | 60.1 ± 0.4 | -1.1 | 60.6 ± 0.5 | -0.6 |
| dcase2023_dev | pAUC source | 62.7 ± 1.3 | 63.9 ± 0.4 | +1.2 | 62.5 ± 0.7 | -0.2 |
| dcase2023_dev | pAUC target | 56.1 ± 1.4 | 56.1 ± 1.3 | -0.0 | 56.2 ± 1.7 | +0.1 |
| dcase2023_dev_crop_per_step | pAUC source | 62.7 ± 1.3 | 64.4 ± 2.2 | +1.7 | 63.2 ± 2.2 | +0.5 |
| dcase2023_dev_crop_per_step | pAUC target | 56.1 ± 1.4 | 56.1 ± 0.7 | +0.0 | 56.0 ± 1.2 | -0.1 |

