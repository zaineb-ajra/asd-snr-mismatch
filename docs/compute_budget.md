# Compute budget (Phase 5)

Estimate from the timings measured during Phase 4 on one RTX 5000 Ada (the study uses two RTX 5000 Ada GPUs). To be approved before any
run of the study (Phase 6).

## Runs

Per model: main split, 4 training conditions (−6, 0, +6 dB, multi-condition) × 3
seeds = 12 runs; 2×2 sensitivity split, 2 training conditions (0, +6 dB) × 3
seeds = 6 runs, on about 15 % more training clips. The multi-condition model
trains for the step budget of a single-SNR model (protocol), so it costs the same
as a single-SNR run, apart from scoring and safeguards. EAT + kNN has no random
choice: its three seeds are identical, one computation per condition.

## Cost per model (one process on one GPU)

| Model | Measured basis | Per run | 12 main runs | 6 sensitivity runs | Total |
|---|---|---|---|---|---|
| Autoencoder | about 40 s per machine ID (verification runs) | ≈ 10 min (15 IDs) | ≈ 2 h | ≈ 1 h | ≈ 3 GPU-h |
| MobileNetV2 | internal controls at 0 dB: 20 + 17 + 12 + 11 min for the four machine types, scoring included | ≈ 1 h | ≈ 12 h | ≈ 7 h | ≈ 19 GPU-h |
| EAT + kNN | embeddings of every study file already cached; kNN only | < 1 min | < 0.2 h | < 0.1 h | < 0.5 GPU-h |
| Wilkinghoff | 0.076 s per step (DCASE 2022 timing), about 1,400 steps per run, backend and scoring | ≈ 3 min | ≈ 0.6 h | ≈ 0.4 h | ≈ 1 GPU-h |
| **OS-SCL** | **40 s per epoch × 300 epochs (two measurements), safeguard and scoring** | **≈ 3.45 h** | **≈ 41 h** | **≈ 23 h** | **≈ 64 GPU-h** |
| **Total** | | | | | **≈ 88 GPU-h** |

**OS-SCL dominates the budget (about three quarters).** Decided on 2026-09-30: it
is accepted as it is; neither the number of epochs (300, fidelity) nor the number
of seeds (3, protocol) is reduced.

## Wall-clock time on two GPUs

Measured on 2026-10-01: two OS-SCL trainings on the same card take 78.5 s per
epoch each, against 40 s for one alone: **no gain** (× 1.02), the model already
saturates the GPU. Two MobileNetV2 trainings per card ran at almost the speed of
one (DCASE 2022 fidelity runs: 14.4 to 15.4 min per run with two per card, 14.2
min alone): gain of about × 1.9.

| Stage | Processes | GPU-hours | Wall-clock |
|---|---|---|---|
| A. Autoencoder, MobileNetV2, EAT, Wilkinghoff | two per card (four at a time) | ≈ 24 | ≈ 6.5 h |
| B. OS-SCL | one per card (two at a time) | ≈ 64 | ≈ 32 h |
| **Total** | | **≈ 88** | **≈ 39 h** |

Stage A first: four of the five models are complete within the first seven
hours, and the per-run safeguards and result files can be checked before the
long stage. The runs are resumable (a finished run is skipped), launched in tmux,
with one log per queue.

## Not included

Phase 4 validation runs already done (about 20 GPU-hours in total); bootstrap
confidence intervals and aggregation (CPU, Phase 7).
