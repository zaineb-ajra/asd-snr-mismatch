**Code validation only. The MIMII AUC values of the internal controls (training at 0 dB, one seed) are implementation checks; they are not results of the study, not even partial ones.**

## Fidelity check (DCASE 2022 Task 2 development set)

### Retained cells (published std ≤ 3.0; agreement: |difference| ≤ 3.0)

| metric | machine_type | section | published | ours | difference | agrees |
|---|---|---|---|---|---|---|
| AUC (source) | slider | 00 | 87.15 ± 2.71 | 84.55 ± 6.49 | -2.60 | yes |
| AUC (source) | valve | 02 | 76.26 ± 1.02 | 77.16 ± 1.01 | +0.90 | yes |
| AUC (target) | valve | 02 | 78.74 ± 2.64 | 79.75 ± 1.83 | +1.01 | yes |
| pAUC | ToyCar | 00 | 49.96 ± 2.56 | 50.42 ± 0.72 | +0.46 | yes |
| pAUC | ToyCar | 01 | 50.92 ± 2.52 | 51.59 ± 1.69 | +0.67 | yes |
| pAUC | ToyTrain | 00 | 50.25 ± 1.49 | 49.76 ± 0.41 | -0.49 | yes |
| pAUC | fan | 02 | 65.14 ± 1.09 | 65.11 ± 0.54 | -0.04 | yes |
| pAUC | slider | 01 | 48.21 ± 2.73 | 48.23 ± 0.67 | +0.02 | yes |
| pAUC | slider | 02 | 49.69 ± 1.63 | 49.32 ± 0.58 | -0.37 | yes |
| pAUC | valve | 02 | 85.74 ± 0.08 | 85.48 ± 0.65 | -0.26 | yes |

10 of 10 retained cells agree.

### Section (ID) classification accuracy on the normal test clips

| machine_type | seeds | accuracy | minutes |
|---|---|---|---|
| ToyCar | 5 | 0.546 ± 0.014 | 14.4 |
| ToyTrain | 5 | 0.387 ± 0.039 | 15.4 |
| fan | 5 | 0.995 ± 0.006 | 14.7 |
| slider | 5 | 0.989 ± 0.007 | 14.4 |
| valve | 5 | 1.000 ± 0.000 | 14.5 |

## Control run: ToyCar, seed 0, validation_split 0.1 vs 0.0

| section | validation_split | ID accuracy (test normals) | AUC (source) | AUC (target) | pAUC |
|---|---|---|---|---|---|
| 00 | 0.0 | 0.75 | 46.74 | 65.20 | 51.42 |
| 00 | 0.1 | 0.75 | 47.94 | 59.52 | 51.32 |
| 01 | 0.0 | 0.46 | 66.34 | 55.76 | 53.11 |
| 01 | 0.1 | 0.65 | 63.36 | 51.78 | 54.32 |
| 02 | 0.0 | 0.85 | 62.72 | 85.16 | 62.05 |
| 02 | 0.1 | 0.25 | 77.80 | 40.20 | 62.11 |

Fidelity runs on ToyCar (validation_split 0.1, 5 seeds, mean ± std):

| section | AUC (source) | AUC (target) | pAUC |
|---|---|---|---|
| 00 | 47.96 ± 2.94 | 56.91 ± 4.08 | 50.42 ± 0.72 |
| 01 | 59.26 ± 4.56 | 54.30 ± 3.47 | 51.59 ± 1.69 |
| 02 | 83.14 ± 3.27 | 44.42 ± 8.93 | 60.36 ± 6.12 |

## Internal controls (study data, main split)

Training condition 0dB, seed 0, commit `4f89867907ef156bcbd9f3ee91a63083e59e5fe8`. Determinism (slider retrained with the same seed, identical scores): yes.

### Training and ID accuracy

| machine_type | train loss (first → last epoch) | train accuracy | ID accuracy, test -6 dB | ID accuracy, test +0 dB | ID accuracy, test +6 dB |
|---|---|---|---|---|---|
| fan | 1.174 → 0.001 | 0.460 → 1.000 | 0.914 | 1.000 | 1.000 |
| pump | 1.015 → 0.006 | 0.532 → 0.998 | 0.900 | 1.000 | 1.000 |
| slider | 1.280 → 0.109 | 0.419 → 0.954 | 0.618 | 0.861 | 0.980 |
| valve | 1.102 → 0.129 | 0.391 → 0.940 | 0.983 | 1.000 | 1.000 |

### Confusion matrices (normal test clips; rows true, columns predicted)

**fan, test -6 dB**

| true | id_00 | id_02 | id_04 | id_06 |
|---|---|---|---|---|
| id_00 | 303 | 21 | 33 | 0 |
| id_02 | 0 | 335 | 0 | 0 |
| id_04 | 14 | 39 | 292 | 1 |
| id_06 | 0 | 0 | 0 | 214 |

**fan, test +0 dB**

| true | id_00 | id_02 | id_04 | id_06 |
|---|---|---|---|---|
| id_00 | 357 | 0 | 0 | 0 |
| id_02 | 0 | 335 | 0 | 0 |
| id_04 | 0 | 0 | 346 | 0 |
| id_06 | 0 | 0 | 0 | 214 |

**fan, test +6 dB**

| true | id_00 | id_02 | id_04 | id_06 |
|---|---|---|---|---|
| id_00 | 357 | 0 | 0 | 0 |
| id_02 | 0 | 335 | 0 | 0 |
| id_04 | 0 | 0 | 346 | 0 |
| id_06 | 0 | 0 | 0 | 214 |

**pump, test -6 dB**

| true | id_00 | id_02 | id_04 | id_06 |
|---|---|---|---|---|
| id_00 | 108 | 0 | 1 | 11 |
| id_02 | 11 | 90 | 1 | 5 |
| id_04 | 1 | 0 | 91 | 0 |
| id_06 | 10 | 0 | 1 | 78 |

**pump, test +0 dB**

| true | id_00 | id_02 | id_04 | id_06 |
|---|---|---|---|---|
| id_00 | 120 | 0 | 0 | 0 |
| id_02 | 0 | 107 | 0 | 0 |
| id_04 | 0 | 0 | 92 | 0 |
| id_06 | 0 | 0 | 0 | 89 |

**pump, test +6 dB**

| true | id_00 | id_02 | id_04 | id_06 |
|---|---|---|---|---|
| id_00 | 120 | 0 | 0 | 0 |
| id_02 | 0 | 107 | 0 | 0 |
| id_04 | 0 | 0 | 92 | 0 |
| id_06 | 0 | 0 | 0 | 89 |

**slider, test -6 dB**

| true | id_00 | id_02 | id_04 | id_06 |
|---|---|---|---|---|
| id_00 | 41 | 312 | 0 | 3 |
| id_02 | 0 | 231 | 1 | 2 |
| id_04 | 0 | 0 | 163 | 1 |
| id_06 | 0 | 0 | 0 | 81 |

**slider, test +0 dB**

| true | id_00 | id_02 | id_04 | id_06 |
|---|---|---|---|---|
| id_00 | 240 | 115 | 0 | 1 |
| id_02 | 0 | 234 | 0 | 0 |
| id_04 | 0 | 0 | 164 | 0 |
| id_06 | 0 | 0 | 0 | 81 |

**slider, test +6 dB**

| true | id_00 | id_02 | id_04 | id_06 |
|---|---|---|---|---|
| id_00 | 340 | 16 | 0 | 0 |
| id_02 | 1 | 233 | 0 | 0 |
| id_04 | 0 | 0 | 164 | 0 |
| id_06 | 0 | 0 | 0 | 81 |

**valve, test -6 dB**

| true | id_02 | id_04 | id_06 |
|---|---|---|---|
| id_02 | 105 | 4 | 0 |
| id_04 | 0 | 94 | 0 |
| id_06 | 1 | 0 | 92 |

**valve, test +0 dB**

| true | id_02 | id_04 | id_06 |
|---|---|---|---|
| id_02 | 109 | 0 | 0 |
| id_04 | 0 | 94 | 0 |
| id_06 | 0 | 0 | 93 |

**valve, test +6 dB**

| true | id_02 | id_04 | id_06 |
|---|---|---|---|
| id_02 | 109 | 0 | 0 |
| id_04 | 0 | 94 | 0 |
| id_06 | 0 | 0 | 93 |

### AUC per machine ID (one seed; implementation check, not a result)

| machine_type | machine_id | test -6 dB | test +0 dB | test +6 dB |
|---|---|---|---|---|
| fan | id_00 | 0.440 | 0.568 | 0.517 |
| fan | id_02 | 0.921 | 0.993 | 0.999 |
| fan | id_04 | 0.584 | 0.714 | 0.826 |
| fan | id_06 | 0.622 | 0.618 | 0.864 |
| pump | id_00 | 0.849 | 0.960 | 0.986 |
| pump | id_02 | 0.449 | 0.712 | 0.844 |
| pump | id_04 | 0.991 | 1.000 | 1.000 |
| pump | id_06 | 0.648 | 0.829 | 0.749 |
| slider | id_00 | 0.968 | 0.999 | 1.000 |
| slider | id_02 | 0.906 | 0.945 | 0.968 |
| slider | id_04 | 0.940 | 0.975 | 1.000 |
| slider | id_06 | 0.590 | 0.759 | 0.730 |
| valve | id_02 | 0.349 | 0.411 | 0.457 |
| valve | id_04 | 0.927 | 0.965 | 0.988 |
| valve | id_06 | 0.810 | 0.863 | 0.865 |

