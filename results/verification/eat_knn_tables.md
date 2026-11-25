**Code validation only. The AUC values below (reference set at 0 dB, including the sensitivity analysis) are implementation checks; they are not results of the study, not even partial ones, and must not be cited as such.**

## (a) Fidelity of the encoder port

90 of 90 clips pass (criterion: relative error <= 0.0001, cosine >= 0.99999); commit `b9a82d4`; reference: transformers 4.51.3, torch 2.11.0+cu128.

| Array | max relative error | min cosine |
|---|---|---|
| fbank | 0.00e+00 | — |
| time_mean | 5.26e-06 | 0.999999999877 |
| cls | 2.72e-05 | 0.999999998666 |

## (c) Internal controls (main split, reference set at 0 dB)

Determinism (valve +0 dB (2934 clips), recomputed with a fresh extractor, bitwise identical): yes. Largest self-distance of a training clip: 4.7e-15.

### Main configuration (temporal mean, K = 1): AUC per machine ID

| machine_type | machine_id | test -6 dB | test +0 dB | test +6 dB |
|---|---|---|---|---|
| fan | id_00 | 0.592 | 0.694 | 0.858 |
| fan | id_02 | 0.955 | 0.992 | 1.000 |
| fan | id_04 | 0.745 | 0.963 | 0.992 |
| fan | id_06 | 0.846 | 0.978 | 1.000 |
| pump | id_00 | 0.775 | 0.919 | 0.961 |
| pump | id_02 | 0.870 | 0.947 | 0.979 |
| pump | id_04 | 0.992 | 1.000 | 1.000 |
| pump | id_06 | 0.876 | 0.984 | 0.985 |
| slider | id_00 | 0.906 | 0.983 | 0.970 |
| slider | id_02 | 0.877 | 0.971 | 0.994 |
| slider | id_04 | 0.802 | 0.882 | 0.971 |
| slider | id_06 | 0.662 | 0.906 | 0.965 |
| valve | id_02 | 0.570 | 0.630 | 0.623 |
| valve | id_04 | 0.660 | 0.748 | 0.675 |
| valve | id_06 | 0.605 | 0.561 | 0.597 |

### Sensitivity analysis (secondary; never replaces the main configuration)

Mean AUC over the 15 machine IDs (and difference with the main configuration), per test SNR:

| variant | test -6 dB | test +0 dB | test +6 dB |
|---|---|---|---|
| cls, K = 1 | 0.736 (-0.046) | 0.843 (-0.034) | 0.863 (-0.041) |
| cls, K = 5 | 0.727 (-0.056) | 0.822 (-0.055) | 0.853 (-0.052) |
| time_mean, K = 1 (main) | 0.782 | 0.877 | 0.905 |
| time_mean, K = 5 | 0.755 (-0.028) | 0.857 (-0.020) | 0.895 (-0.010) |

Spread of the AUC across the four variants, per (machine ID, test SNR), 45 cells: median 0.084, maximum 0.238.

