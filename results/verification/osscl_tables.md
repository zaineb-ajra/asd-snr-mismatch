**Code validation only. The AUC values below (training at 0 dB, one seed) are implementation checks; they are not results of the study, not even partial ones.**

## Internal controls (study data, main split)

Training condition 0dB, seed 0, commit `8bcc16b01adfe14c220933cf19812a4689a36dec`. Determinism ( retrained with the same seed, identical scores): yes.

### Training and ID accuracy

| machine_type | train loss (first → last epoch) | train accuracy | ID accuracy, test -6 dB | ID accuracy, test +0 dB | ID accuracy, test +6 dB |
|---|---|---|---|---|---|
| all | 27.673 → 7.519 | — | 0.767 | 0.992 | 1.000 |

### Confusion matrices (normal test clips; rows true, columns predicted)

**all, test -6 dB**

| true | fan/id_00 | fan/id_02 | fan/id_04 | fan/id_06 | pump/id_00 | pump/id_02 | pump/id_04 | pump/id_06 | slider/id_00 | slider/id_02 | slider/id_04 | slider/id_06 | valve/id_02 | valve/id_04 | valve/id_06 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fan/id_00 | 302 | 14 | 13 | 0 | 25 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| fan/id_02 | 0 | 335 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| fan/id_04 | 0 | 0 | 343 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| fan/id_06 | 0 | 0 | 2 | 212 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pump/id_00 | 2 | 2 | 4 | 0 | 112 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pump/id_02 | 1 | 2 | 1 | 0 | 6 | 97 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pump/id_04 | 0 | 0 | 0 | 0 | 2 | 0 | 90 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pump/id_06 | 7 | 8 | 3 | 0 | 10 | 0 | 0 | 61 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| slider/id_00 | 35 | 13 | 58 | 3 | 187 | 6 | 0 | 12 | 1 | 40 | 1 | 0 | 0 | 0 | 0 |
| slider/id_02 | 5 | 2 | 10 | 0 | 52 | 2 | 0 | 3 | 0 | 160 | 0 | 0 | 0 | 0 | 0 |
| slider/id_04 | 0 | 1 | 4 | 0 | 7 | 3 | 0 | 0 | 0 | 2 | 147 | 0 | 0 | 0 | 0 |
| slider/id_06 | 0 | 6 | 0 | 0 | 11 | 3 | 0 | 1 | 0 | 1 | 0 | 59 | 0 | 0 | 0 |
| valve/id_02 | 1 | 17 | 6 | 0 | 12 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 67 | 2 | 1 |
| valve/id_04 | 2 | 2 | 2 | 0 | 15 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 71 | 0 |
| valve/id_06 | 1 | 2 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 85 |

**all, test +0 dB**

| true | fan/id_00 | fan/id_02 | fan/id_04 | fan/id_06 | pump/id_00 | pump/id_02 | pump/id_04 | pump/id_06 | slider/id_00 | slider/id_02 | slider/id_04 | slider/id_06 | valve/id_02 | valve/id_04 | valve/id_06 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fan/id_00 | 357 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| fan/id_02 | 0 | 335 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| fan/id_04 | 0 | 0 | 346 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| fan/id_06 | 0 | 0 | 0 | 214 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pump/id_00 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pump/id_02 | 0 | 0 | 0 | 0 | 0 | 107 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pump/id_04 | 0 | 0 | 0 | 0 | 0 | 0 | 92 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pump/id_06 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 89 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| slider/id_00 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 335 | 19 | 0 | 0 | 0 | 0 | 0 |
| slider/id_02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 234 | 0 | 0 | 0 | 0 | 0 |
| slider/id_04 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 164 | 0 | 0 | 0 | 0 |
| slider/id_06 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 81 | 0 | 0 | 0 |
| valve/id_02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 109 | 0 | 0 |
| valve/id_04 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 94 | 0 |
| valve/id_06 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 93 |

**all, test +6 dB**

| true | fan/id_00 | fan/id_02 | fan/id_04 | fan/id_06 | pump/id_00 | pump/id_02 | pump/id_04 | pump/id_06 | slider/id_00 | slider/id_02 | slider/id_04 | slider/id_06 | valve/id_02 | valve/id_04 | valve/id_06 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fan/id_00 | 357 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| fan/id_02 | 0 | 335 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| fan/id_04 | 0 | 0 | 346 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| fan/id_06 | 0 | 0 | 0 | 214 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pump/id_00 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pump/id_02 | 0 | 0 | 0 | 0 | 0 | 107 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pump/id_04 | 0 | 0 | 0 | 0 | 0 | 0 | 92 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pump/id_06 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 89 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| slider/id_00 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 356 | 0 | 0 | 0 | 0 | 0 | 0 |
| slider/id_02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 234 | 0 | 0 | 0 | 0 | 0 |
| slider/id_04 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 164 | 0 | 0 | 0 | 0 |
| slider/id_06 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 81 | 0 | 0 | 0 |
| valve/id_02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 109 | 0 | 0 |
| valve/id_04 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 94 | 0 |
| valve/id_06 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 93 |

### AUC per machine ID (one seed; implementation check, not a result)

| machine_type | machine_id | test -6 dB | test +0 dB | test +6 dB |
|---|---|---|---|---|
| fan | id_00 | 0.361 | 0.503 | 0.581 |
| fan | id_02 | 0.863 | 0.994 | 0.999 |
| fan | id_04 | 0.586 | 0.821 | 0.864 |
| fan | id_06 | 0.610 | 0.721 | 0.747 |
| pump | id_00 | 0.828 | 0.970 | 0.988 |
| pump | id_02 | 0.512 | 0.745 | 0.799 |
| pump | id_04 | 0.991 | 0.998 | 1.000 |
| pump | id_06 | 0.909 | 0.988 | 0.994 |
| slider | id_00 | 0.804 | 0.997 | 1.000 |
| slider | id_02 | 0.773 | 0.988 | 1.000 |
| slider | id_04 | 0.900 | 0.990 | 1.000 |
| slider | id_06 | 0.922 | 0.997 | 1.000 |
| valve | id_02 | 0.295 | 0.371 | 0.369 |
| valve | id_04 | 0.829 | 0.986 | 0.987 |
| valve | id_06 | 0.957 | 0.997 | 0.999 |

