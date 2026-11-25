**Superseded: internal controls with the repository EMA decay 0.9999 (run of 2026-09-30), kept as evidence of the problem described in docs/deviations.md (O-EMA). The current controls are in osscl_tables.md.**

**Code validation only. The AUC values below (training at 0 dB, one seed) are implementation checks; they are not results of the study, not even partial ones.**

## Internal controls (study data, main split)

Training condition 0dB, seed 0, commit `1661f1053bf5d6f9b7ab0e93f0a77b2153a84e64`. Determinism ( retrained with the same seed, identical scores): yes.

### Training and ID accuracy

| machine_type | train loss (first → last epoch) | train accuracy | ID accuracy, test -6 dB | ID accuracy, test +0 dB | ID accuracy, test +6 dB |
|---|---|---|---|---|---|
| all | 27.673 → 7.519 | — | 0.329 | 0.858 | 0.912 |

### Confusion matrices (normal test clips; rows true, columns predicted)

**all, test -6 dB**

| true | fan/id_00 | fan/id_02 | fan/id_04 | fan/id_06 | pump/id_00 | pump/id_02 | pump/id_04 | pump/id_06 | slider/id_00 | slider/id_02 | slider/id_04 | slider/id_06 | valve/id_02 | valve/id_04 | valve/id_06 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fan/id_00 | 149 | 2 | 0 | 0 | 182 | 0 | 0 | 14 | 0 | 10 | 0 | 0 | 0 | 0 | 0 |
| fan/id_02 | 0 | 137 | 0 | 0 | 196 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| fan/id_04 | 1 | 0 | 50 | 0 | 277 | 0 | 0 | 12 | 0 | 6 | 0 | 0 | 0 | 0 | 0 |
| fan/id_06 | 0 | 0 | 0 | 66 | 131 | 0 | 0 | 11 | 0 | 6 | 0 | 0 | 0 | 0 | 0 |
| pump/id_00 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pump/id_02 | 0 | 0 | 0 | 0 | 48 | 59 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pump/id_04 | 0 | 0 | 0 | 0 | 45 | 0 | 47 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pump/id_06 | 0 | 0 | 0 | 0 | 12 | 0 | 0 | 77 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| slider/id_00 | 4 | 0 | 0 | 0 | 270 | 0 | 0 | 13 | 27 | 41 | 0 | 0 | 0 | 1 | 0 |
| slider/id_02 | 1 | 0 | 0 | 0 | 119 | 0 | 0 | 1 | 3 | 110 | 0 | 0 | 0 | 0 | 0 |
| slider/id_04 | 0 | 0 | 0 | 0 | 138 | 0 | 0 | 0 | 0 | 1 | 25 | 0 | 0 | 0 | 0 |
| slider/id_06 | 0 | 0 | 0 | 0 | 36 | 0 | 0 | 1 | 0 | 0 | 0 | 44 | 0 | 0 | 0 |
| valve/id_02 | 0 | 0 | 0 | 0 | 105 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| valve/id_04 | 0 | 0 | 0 | 0 | 87 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 0 |
| valve/id_06 | 0 | 0 | 0 | 0 | 92 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |

**all, test +0 dB**

| true | fan/id_00 | fan/id_02 | fan/id_04 | fan/id_06 | pump/id_00 | pump/id_02 | pump/id_04 | pump/id_06 | slider/id_00 | slider/id_02 | slider/id_04 | slider/id_06 | valve/id_02 | valve/id_04 | valve/id_06 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fan/id_00 | 345 | 0 | 0 | 0 | 1 | 0 | 0 | 9 | 0 | 1 | 0 | 0 | 0 | 1 | 0 |
| fan/id_02 | 0 | 333 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| fan/id_04 | 0 | 0 | 201 | 0 | 35 | 0 | 0 | 95 | 3 | 10 | 0 | 0 | 0 | 2 | 0 |
| fan/id_06 | 0 | 0 | 0 | 214 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pump/id_00 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pump/id_02 | 0 | 0 | 0 | 0 | 0 | 107 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pump/id_04 | 0 | 0 | 0 | 0 | 0 | 0 | 92 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pump/id_06 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 89 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| slider/id_00 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 353 | 2 | 0 | 0 | 0 | 0 | 0 |
| slider/id_02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 43 | 191 | 0 | 0 | 0 | 0 | 0 |
| slider/id_04 | 0 | 0 | 0 | 0 | 11 | 0 | 0 | 0 | 0 | 5 | 147 | 1 | 0 | 0 | 0 |
| slider/id_06 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 80 | 0 | 0 | 0 |
| valve/id_02 | 0 | 0 | 0 | 0 | 20 | 0 | 0 | 27 | 3 | 0 | 0 | 35 | 0 | 24 | 0 |
| valve/id_04 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 91 | 0 |
| valve/id_06 | 0 | 0 | 0 | 0 | 50 | 1 | 0 | 4 | 0 | 5 | 0 | 1 | 0 | 0 | 32 |

**all, test +6 dB**

| true | fan/id_00 | fan/id_02 | fan/id_04 | fan/id_06 | pump/id_00 | pump/id_02 | pump/id_04 | pump/id_06 | slider/id_00 | slider/id_02 | slider/id_04 | slider/id_06 | valve/id_02 | valve/id_04 | valve/id_06 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fan/id_00 | 357 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| fan/id_02 | 0 | 335 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| fan/id_04 | 0 | 0 | 226 | 0 | 1 | 0 | 0 | 63 | 49 | 1 | 0 | 0 | 0 | 6 | 0 |
| fan/id_06 | 0 | 0 | 0 | 214 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pump/id_00 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pump/id_02 | 0 | 0 | 0 | 0 | 0 | 107 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pump/id_04 | 0 | 0 | 0 | 0 | 0 | 0 | 92 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pump/id_06 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 89 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| slider/id_00 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 356 | 0 | 0 | 0 | 0 | 0 | 0 |
| slider/id_02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 12 | 222 | 0 | 0 | 0 | 0 | 0 |
| slider/id_04 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 163 | 0 | 0 | 0 | 0 |
| slider/id_06 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 81 | 0 | 0 | 0 |
| valve/id_02 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 6 | 24 | 0 | 0 | 34 | 0 | 37 | 3 |
| valve/id_04 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 94 | 0 |
| valve/id_06 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 88 |

### AUC per machine ID (one seed; implementation check, not a result)

| machine_type | machine_id | test -6 dB | test +0 dB | test +6 dB |
|---|---|---|---|---|
| fan | id_00 | 0.392 | 0.471 | 0.574 |
| fan | id_02 | 0.856 | 0.994 | 0.997 |
| fan | id_04 | 0.572 | 0.706 | 0.839 |
| fan | id_06 | 0.710 | 0.753 | 0.676 |
| pump | id_00 | 0.604 | 0.586 | 0.656 |
| pump | id_02 | 0.588 | 0.708 | 0.844 |
| pump | id_04 | 0.976 | 0.995 | 0.999 |
| pump | id_06 | 0.971 | 0.975 | 0.964 |
| slider | id_00 | 0.948 | 0.996 | 1.000 |
| slider | id_02 | 0.831 | 0.823 | 0.896 |
| slider | id_04 | 0.853 | 0.958 | 0.992 |
| slider | id_06 | 0.876 | 0.982 | 0.990 |
| valve | id_02 | 0.498 | 0.517 | 0.466 |
| valve | id_04 | 0.555 | 0.678 | 0.723 |
| valve | id_06 | 0.532 | 0.846 | 0.952 |

