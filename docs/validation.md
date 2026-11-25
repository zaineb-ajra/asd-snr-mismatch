# Validation of the models (not part of the study)

Each model is checked before its study runs. The scope and the criteria of each
check are fixed and committed before the check is run; its results are in
`results/verification/`. Validation data other than MIMII are never used in the
study.

## Autoencoder

Published AUCs of the MIMII paper (Table 3), reproduced with the original MIMII
split; see `results/verification/ae_tables.md` and deviation A6.

## MobileNetV2 (DCASE 2022 Task 2 baseline)

Scope decided on 2026-09-30: (1) fidelity check against the published results of
the ported code, and (3) internal controls. The comparison with other
implementations on DCASE 2020 is not made (a difference would not be
interpretable, an agreement would prove nothing).

**Published reference.** Dohi et al. 2022, Table 2: MobileNetV2 baseline on the
DCASE 2022 Task 2 development set (MIMII DG + ToyADMOS2; Zenodo 6355122, CC BY 4.0,
not derived from MIMII), mean ± standard deviation over 5 trials, per section:
AUC (source), AUC (target), pAUC. Transcribed in
[`metadata/published/dohi2022_table2_mobilenetv2.csv`](../metadata/published/dohi2022_table2_mobilenetv2.csv).
Note: the bearing rows of Table 2 are identical to those of Table 1
(autoencoder), a likely copy error in the paper; no bearing cell passes the
selection below anyway.

**Metrics, as in the baseline code** (`01_test.py`, commit `aaf3932`): AUC (source)
over the source-domain normal clips and all anomalous clips of the section; AUC
(target) likewise with target-domain normal clips; pAUC over all clips of the
section, `roc_auc_score(..., max_fpr=0.1)`.

**Selection of the compared cells (fixed before any run).** A (section, metric)
cell is compared if its published standard deviation is at most **3.0 points**.
Applied mechanically (`tests/test_validation_criteria.py`):

| Metric | Retained cells (published mean ± std) |
|---|---|
| AUC (source) | slider 00 (87.15 ± 2.71), valve 02 (76.26 ± 1.02) |
| AUC (target) | valve 02 (78.74 ± 2.64) |
| pAUC | ToyCar 00 (49.96 ± 2.56), ToyCar 01 (50.92 ± 2.52), ToyTrain 00 (50.25 ± 1.49), fan 02 (65.14 ± 1.09), slider 01 (48.21 ± 2.73), slider 02 (49.69 ± 1.63), valve 02 (85.74 ± 0.08) |

Every metric keeps at least one cell. Six of the ten retained cells are pAUCs,
four of them near chance level (≈ 50), which constrains the implementation less
than the others.

**Agreement rule (fixed before any run).** A retained cell agrees if the mean of
our 5 runs (5 seeds, as the 5 published trials) is within **3.0 points** of the
published mean. Each result is reported with its difference and our standard
deviation; no cell is added or removed after the runs.

**Run settings.** The original configuration, including Keras
`validation_split = 0.1` (the study uses 0.0, deviation A6); one model per
machine type, trained on its three sections. Only the machine types with a
retained cell are downloaded and run: ToyCar, ToyTrain, fan, slider, valve
(4.5 GB of the 6.2 GB).

**Control run (added on 2026-09-30, before it was run).** ToyCar, seed 0,
`validation_split = 0.0`, otherwise identical to the fidelity runs
(`--validation-split 0.0`, outputs in `dcase2022_validation_split_0.0/`). Purpose:
show the effect of the held-out block (deviation M5) and separate a defect of the
baseline from a defect of the port. In the original configuration the held-out
block is, for every machine type, the last 300 training clips of section 02 in
file-name order: 290 source-domain clips and all 10 target-domain clips
(computed from the file order, not from a run); section 02 is therefore trained
without any target-domain clip. Reading, fixed now: the run is compared with
seed 0 of the fidelity check on ToyCar; a change concentrated on section 02 (ID
accuracy of its normal test clips, AUC target), with sections 00 and 01 within
the seed-to-seed spread of the five fidelity runs, attributes the behaviour of
section 02 to the held-out block. With one seed, differences are reported, not
tested.

**Internal controls** (on the study data, MIMII): accuracy and confusion matrix of
the auxiliary machine-ID classification per machine type, computed on the normal
test clips (held-out normal clips of known ID); training loss and accuracy
curves; distributions of the anomaly scores of normal and anomalous clips;
determinism.

### MobileNetV2 results (runs of 2026-09-30)

Tables: [`results/verification/mobilenetv2_tables.md`](../results/verification/mobilenetv2_tables.md);
per-cell values: `results/verification/mobilenetv2_dcase2022_summary.csv` and
`mobilenetv2_dcase2022_comparison.csv`. 25 fidelity runs (5 types × 5 seeds,
about 14.5 min each), 1 control run, 1 internal-control run; no failure.

**Fidelity: 10 of the 10 retained cells agree** (|difference| ≤ 3.0 points):

| Metric | Cell | Published | Ours (5 seeds) | Difference |
|---|---|---|---|---|
| AUC (source) | slider 00 | 87.15 ± 2.71 | 84.55 ± 6.49 | −2.60 |
| AUC (source) | valve 02 | 76.26 ± 1.02 | 77.16 ± 1.01 | +0.90 |
| AUC (target) | valve 02 | 78.74 ± 2.64 | 79.75 ± 1.83 | +1.01 |
| pAUC | ToyCar 00 | 49.96 ± 2.56 | 50.42 ± 0.72 | +0.46 |
| pAUC | ToyCar 01 | 50.92 ± 2.52 | 51.59 ± 1.69 | +0.67 |
| pAUC | ToyTrain 00 | 50.25 ± 1.49 | 49.76 ± 0.41 | −0.49 |
| pAUC | fan 02 | 65.14 ± 1.09 | 65.11 ± 0.54 | −0.04 |
| pAUC | slider 01 | 48.21 ± 2.73 | 48.23 ± 0.67 | +0.02 |
| pAUC | slider 02 | 49.69 ± 1.63 | 49.32 ± 0.58 | −0.37 |
| pAUC | valve 02 | 85.74 ± 0.08 | 85.48 ± 0.65 | −0.26 |

**Unexplained discrepancy, outside the retained cells (reported, not hidden).**
fan section 01, AUC (source): **ours 62.5 ± 5.4, published 76.3 ± 4.95**, a
difference of −13.7 points, i.e. **−2.8 published standard deviations**. We have
no explanation for it. The cell is not among the retained cells because its
published standard deviation exceeds 3.0; this selection criterion was fixed and
committed before any run (commit `543e4fc`, `tests/test_validation_criteria.py`)
and was not revised after the results. The discrepancy was not investigated
further (decision of 2026-09-30). The classifier itself works on this machine
type: section classification accuracy on the fan normal test clips is 0.995 ±
0.006.

*All 45 cells, for information* (differences in units of the published standard
deviation): 41 within 1, 43 within 2. Besides fan 01 AUC (source), the only cell
beyond 2 is valve 02 pAUC (−0.26 points, published std 0.08). fan 00 AUC
(target) is also low (42.0 ± 9.8 vs 62.1 ± 12.5, −1.6).

**Control run (validation_split 0.0, ToyCar, seed 0), read as fixed above.**

| Section 02 | 0.1 (5 seeds: mean ± std, [min, max]) | 0.0 (seed 0) |
|---|---|---|
| ID accuracy, normal test clips | 0.23 [0.21, 0.25] | 0.85 |
| AUC (target) | 44.42 ± 8.92 [36.60, 55.14] | 85.16 (+4.6 std) |
| AUC (source) | 83.14 ± 3.27 [77.80, 86.18] | 62.72 (−6.2 std) |
| pAUC | 60.36 ± 6.12 | 62.05 (+0.3 std) |

Sections 00 and 01 change much less (|z| ≤ 2.0 for their six AUC/pAUC values),
but not all stay inside the five-seed range: section 00 AUC (target) 65.20 (max
61.38, z = +2.0), section 01 AUC (source) 66.34 (max 63.36, z = +1.6), section 01
ID accuracy 0.46 ([0.58, 0.69]). The rule fixed before the run ("within the
seed-to-seed spread") did not say which spread; by the five-seed range it is met
by four of the six AUC/pAUC values of sections 00–01. With validation_split 0.0
the classes are also rebalanced (section 02 trains on 1,000 clips instead of
700), which can move the other sections; one seed does not separate the two.

Mechanism, seed 0 (scores in `scores.csv`): with 0.1, the section-02
target-domain clips, never trained on, receive high anomaly scores (median:
source normal 0.67, target normal 3.21, target anomalous 4.84). AUC (target)
collapses because target normals score high; AUC (source), computed with all
anomalous clips of the section, is inflated by the target anomalies (source
normals vs source anomalies 59.7, vs target anomalies 95.9). With 0.0: 61.9 and
63.6. Conclusion: the held-out block of the original configuration (deviation
M5) drives the section-02 results; our section-02 numbers with 0.1 reproduce the
published ones (AUC target 44.42 vs 45.64; with 0.0: 85.16), so the published
section-02 values carry this property of the baseline, and the port reproduces it
rather than introducing it.

**Internal controls** (MIMII main split, training at 0 dB, seed 0; tables in
`mobilenetv2_tables.md`). Machine-ID accuracy on the normal test clips:

| Machine type | test −6 dB | test 0 dB | test +6 dB | final training accuracy |
|---|---|---|---|---|
| fan | 0.914 | 1.000 | 1.000 | 1.000 |
| pump | 0.900 | 1.000 | 1.000 | 0.998 |
| slider | 0.618 | 0.861 | 0.980 | 0.954 |
| valve | 0.983 | 1.000 | 1.000 | 0.940 |

No class is confused at the training SNR except slider `id_00`, of which 115 of
356 normal test clips are predicted `id_02` at 0 dB (312 at −6 dB, 16 at +6 dB).
Slider `id_00` has 178 training clips against 708 for `id_02`, an imbalance created
by our identity exclusions, not present in the MIMII release (protocol, "Known
limitations"); the confusion goes to the majority class and fades
at higher SNR, which points to class imbalance and noise, not to a labelling
error. Errors elsewhere appear only at −6 dB, below the training SNR. Training loss decreases at every epoch for pump, slider and valve; for fan it
rises once, at epoch 16 (0.0013 to 0.0017, near zero). Determinism: slider retrained with
the same seed gives identical scores.

## EAT + kNN

Scope decided on 2026-09-30: (a) fidelity of the encoder port and (c) internal
controls. Not done: (b) reproducing a published frozen-EAT kNN result on DCASE
2020 Task 2, for two reasons. First, a difference would not be interpretable:
the two published values for the same checkpoint on that set differ by 5.7
points (77.8, Saengthong & Shinozaki 2026; 72.1, Zhang et al. 2025, ECHO), and
both are aggregates over six machine types that depend on undocumented or
differing details (pooling, K, aggregation). Second, the comparison would need
the ToyADMOS part of that set (ToyCar, ToyConveyor), distributed under CC
BY-NC-SA 4.0 (non-commercial), which this study does not use. This data is not downloaded. No published per-machine result of
frozen EAT with kNN on MIMII-derived data was found.

**(a) Fidelity of the encoder port (criterion fixed on 2026-09-30, before the
run).**

- *Reference:* the official inference path of the model card at the pinned
  revision: the downloaded snapshot (`model_core.py`, `eat_model.py`,
  `modeling_eat.py`, `model.safetensors`) loaded with `transformers`
  (`AutoModel.from_pretrained(<local snapshot>, trust_remote_code=True)`, local
  files only) in a separate environment, never in the study code; audio read
  from a mono WAV file with `soundfile`, preprocessing copied from the model
  card; one clip per forward pass.
- *Ours:* the study pipeline (`noisemismatch/models/eat_knn.py`): audio from the
  channel-0 cache, vendored encoder, batched forward passes.
- *Clips:* 90: for each of the 15 study machine IDs, each SNR, one normal and one
  anomalous test clip of the main split, drawn with seed 2026. The reference
  reads channel 0 of each clip written to a mono 16-bit WAV file.
- *Both sides:* the same GPU, float32, TF32 disabled.
- *Compared:* the normalized filterbank input (1,024 × 128) and both pooled
  embeddings (temporal mean, 6,144; CLS, 768).
- *Criterion:* for every clip and each of the three arrays, maximum absolute
  difference ≤ 1e-4 × maximum absolute value of the reference, and, for the two
  embeddings, cosine similarity ≥ 0.99999. The check passes only if all 90 clips
  pass; the maximum differences are reported either way.

**(c) Internal controls** (study data, main split, reference set at 0 dB):
determinism (embeddings of one machine type computed twice are identical);
self-distance (a training clip scored against a reference set that contains it
gets distance 0 up to 1e-6); AUC and score quartiles per machine ID and test SNR;
the sensitivity analysis of the protocol for this condition.

### EAT + kNN results (runs of 2026-09-30)

Tables: [`results/verification/eat_knn_tables.md`](../results/verification/eat_knn_tables.md).
Embeddings of all 47,374 cached study files computed once (about 6 min on one
GPU) and cached; everything below reuses them.

**(a) Fidelity: passes, 90 of 90 clips.** The normalized filterbank input is
bitwise identical to the official path (maximum difference 0); largest relative
error of the embeddings 5.3e-6 (temporal mean) and 2.7e-5 (CLS), smallest cosine
similarity 0.99999999988 and 0.9999999987 (criterion: 1e-4 and 0.99999). The
reference ran with `transformers` 4.51.3, the version recorded in the checkpoint's
`config.json`; that version copies into its module cache only the files imported
by `modeling_eat.py`, so the snapshot's Python files were copied there unchanged
before loading (`scripts/eat_reference.py`).

**(c) Internal controls (reference set at 0 dB): pass.** Embeddings of valve at 0
dB (2,934 files) recomputed with a fresh extractor are bitwise identical to the
cached ones; the largest distance of a training clip to its own reference set is
4.7e-15. Main configuration, mean AUC over the 15 machine IDs: 0.782 (test −6
dB), 0.877 (0 dB), 0.905 (+6 dB); per-ID values in the tables. **These AUCs are
implementation checks, not results of the study, not even partial ones; they must
not be reported as such anywhere.**

**Sensitivity analysis, this condition only (secondary).** Mean AUC over the 15
machine IDs, difference with the main configuration at −6 / 0 / +6 dB: CLS, K = 1:
−0.046 / −0.034 / −0.041; CLS, K = 5: −0.056 / −0.055 / −0.052; temporal mean,
K = 5: −0.028 / −0.020 / −0.010. Across the four variants the AUC of a (machine
ID, test SNR) cell varies by 0.084 (median over 45 cells) and up to 0.238. The
main configuration happens to score highest here; it was fixed before any
embedding was computed and would have been kept otherwise.

### Wilkinghoff results (runs of 2026-09-30, 5 seeds per dataset)

Tables: [`results/verification/wilkinghoff_tables.md`](../results/verification/wilkinghoff_tables.md)
(`scripts/wilkinghoff_report.py`). Class counts as the author's (342 and 186).
About 8 to 9 min per seed on one GPU for each dataset. As decided on 2026-10-01,
the pre-registered criterion is evaluated under both per-domain definitions and
both verdicts are reported; neither is chosen after the fact.

| Set | Cell | Published | Ours, official (diff) | Ours, restricted (diff) |
|---|---|---|---|---|
| DCASE 2022 dev | AUC source | 86.0 ± 0.9 | 83.1 ± 1.6 (−2.9) | 85.3 ± 0.8 (−0.7) |
| | AUC target | 78.2 ± 0.7 | 75.1 ± 0.9 (**−3.1**) | 77.5 ± 1.4 (−0.7) |
| | AUC mixed | 82.5 ± 0.8 | 81.7 ± 1.0 (−0.8) | same |
| | pAUC mixed | 68.2 ± 1.1 | 68.0 ± 0.8 (−0.2) | same |
| DCASE 2022 eval | AUC source | 77.7 ± 0.8 | 81.7 ± 1.2 (**+4.0**) | 76.4 ± 0.9 (−1.3) |
| | AUC target | 71.6 ± 1.0 | 58.9 ± 0.7 (**−12.7**) | 68.9 ± 0.5 (−2.7) |
| | AUC mixed | 74.2 ± 0.3 | 72.6 ± 0.5 (−1.6) | same |
| | pAUC mixed | 61.2 ± 0.3 | 60.8 ± 0.7 (−0.4) | same |
| DCASE 2023 dev | AUC source | 71.2 ± 1.6 | 74.4 ± 1.3 (**+3.2**) | 69.1 ± 1.8 (−2.1) |
| | AUC target | 75.0 ± 1.5 | 63.1 ± 2.8 (**−11.9**) | 69.0 ± 2.2 (**−6.0**) |
| | AUC mixed | 73.1 ± 0.9 | 69.2 ± 1.4 (**−3.9**) | same |
| | pAUC mixed | 57.3 ± 0.6 | 57.7 ± 0.7 (+0.4) | same |

**Verdict of the pre-registered criterion** (|difference| ≤ 3.0): **6 of 12 cells
under the official definition** (the definition the criterion was registered
with), **10 of 12 under the restricted definition**. The six cells that do not
depend on the definition (mixed): 5 of 6 agree.

**Observation (not an adjustment of the criterion).** The published per-domain
values follow the restricted definition (clips of the domain only), not the
official DCASE one: on DCASE 2022, the eight per-domain values (AUC and pAUC,
source and target, dev and eval) are within 2.7 points of ours under the
restricted definition, and up to 12.7 points away under the official one (per-domain
pAUC, for information: within 0.7 under the restricted definition). The author's
per-domain numbers are therefore not comparable with numbers computed with the
official definition.

**Disagreement that remains under both definitions: DCASE 2023 dev.** AUC
(target) is 6.0 points below the published value under the restricted definition
(11.9 under the official one) and AUC (mixed) 3.9 below; the pAUC cells agree.
Not explained. Candidates, none tested: the elements present in the author's code
but in none of his articles (squeeze-and-excitation blocks, L2 weight decay; not
implemented under the paper-only rule), the handling of the 6 to 18 s clips
(repeated and cropped once here), other unpublished details. On DCASE 2022 the
re-implementation reproduces every definition-independent cell within 1.6
points.

**Control run on DCASE 2023: random crop at every training step (added on
2026-10-01, before it was run).** Reasoning: the remaining disagreement is
specific to DCASE 2023, whereas elements missing from the articles
(squeeze-and-excitation blocks, L2 weight decay) would also affect DCASE 2022,
which is reproduced within 1.6 points on the definition-independent cells; what
distinguishes DCASE 2023 is the variable clip length (6 to 18 s against 10 s), so
the handling of the length is the most plausible cause. The author's report says
the waveforms are "repeated (and randomly cropped)" to the longest length. Variant
tested: each clip is repeated to the first multiple of its length that reaches
18 s and a crop start is drawn uniformly at every training step; at inference the
repeated clip is taken from its start. Five seeds (`--crop per-step`, outputs in
`dcase2023_crop_per_step/`). Reading, fixed now: if the two cells that disagree
under both definitions (AUC target and AUC mixed on DCASE 2023 dev) come within
3.0 points of the published values and no agreeing cell is lost, the variant is
adopted for the fidelity check and documented as deduced; otherwise the model is
kept as it is and the DCASE 2023 disagreement is documented as a limitation. In
both cases the elements absent from the articles are not added. The study on
MIMII (clips of equal length) is not affected by this choice.

**Result of the control run (2026-10-01, 5 seeds).** Random crop at every
training step does not close the gap on DCASE 2023 dev: AUC (target) 70.0 ± 2.5
under the restricted definition (cropped once: 69.0 ± 2.2; published 75.0 ± 1.5)
and 64.5 ± 2.3 under the official one (63.1 ± 2.8); AUC (mixed) 69.3 ± 1.2 (69.2
± 1.4; published 73.1 ± 0.9). The changes are within the seed-to-seed spread. By
the reading fixed before the run, the variant is not adopted: the model is kept
as it is (clips cropped once), and the squeeze-and-excitation blocks and the L2
weight decay of the author's code are not added (they would also affect DCASE
2022, which is reproduced; adding them would make the model a reconstruction
from the repository).

**Status (decided on 2026-10-01).** The Wilkinghoff re-implementation is validated
as it is, with this limitation: it reproduces the published DCASE 2022 results
(every definition-independent cell within 1.6 points; per-domain cells within
2.7 points under the definition the author uses), but on DCASE 2023 dev it is 5
to 6 points below the published AUC in the target domain and 3.9 below the
domain-independent AUC, a disagreement that the handling of the clip length does
not explain and that remains unexplained.

## OS-SCL

Scope decided on 2026-09-30: unit tests of each component against the formulas
of the paper (noise-supervised contrastive loss computed by hand, ArcFace
margin, Noisy-ArcMix, feature perturbation head, TFgram shapes of Table I) and
internal controls (machine-ID accuracy and confusion per machine type, training
curves, score distributions, determinism). No comparison with the published
numbers: they are reported without dispersion, so no cell can pass a
pre-registered dispersion criterion, and the published model is trained with
non-commercial data (ToyADMOS, DCASE 2020 additional training set). No criterion
is relaxed to allow a comparison.

### OS-SCL: diagnostic on the DCASE 2020 split (decided on 2026-10-05, AFTER the study results)

**Status.** Decided on 2026-10-05, after the results of the study were known:
a dated analysis posterior to the results (protocol freeze rule). It reopens the
decision of 2026-09-30 above (no comparison with the published numbers) for this
diagnostic only. Arms, criterion and consequences are fixed in this note,
committed before any run.
**The results of the main study do not change, whatever the outcome.**

**Why.** Trained and tested at −6 dB, our OS-SCL reaches 87.3 AUC (mean of the
four machine types) against 98.53 for TFSTgram (OS-SCL) as published on the DCASE
2020 development set (Huang et al. 2025, Table II; mean, computed by us, of the
four published per-type values). Our Wilkinghoff reaches 97.6 in the same cell.

**Split.** DCASE 2020 Task 2 development set, MIMII part (fan, pump, slider,
valve; 16 machines, valve `id_00` included), rebuilt from the MIMII −6 dB archives
(channel 0) with `metadata/dcase2020_mimii_mapping.csv` ([`data.md`](data.md)): training
on the 13,119 normal clips of `train/`; test on the 1,600 normal (100 per machine)
and 3,300 anomalous clips of `test/`. Classes: the 16 machine IDs. All training
clips are used, as in the study (the authors keep a random 95 %). Code:
`noisemismatch/dcase2020_check.py`, `configs/dcase2020_check.yaml`.

**Arms (one seed each, seed 0).**

- *A*: the study configuration (`configs/models/osscl.yaml`), code unchanged. The
  options added for B are absent from this configuration; with them absent the
  network, its initialization and its outputs are those of the study runs
  (checked against commit `df3e917`: same parameters at initialization, same
  log-Mel spectrogram and features).
- *B*: the four values of the huangswt/OS-SCL repository recorded as facts on
  2026-09-30, listed in [`deviations.md`](deviations.md) ("OS-SCL, arm B"): log-Mel
  window 2048 in decibels; three MobileFaceNet bottleneck sequences; anomaly
  score = cross-entropy with the margin; EMA decay 0.9999. No code of the
  repository is used. Every other value as in A.
- *Control*: Wilkinghoff, study configuration (16 classes). No criterion.

**Criterion (fixed before the runs).** For A and for B: the mean AUC of the four
machine types (mean over the machines of a type, then over the types) is within
3 points of 98.53, i.e. |mean − 98.53| ≤ 3.00. AUC and pAUC per type and per
machine are reported for every arm (`results/dcase2020_check.md`).

**Consequences (fixed before the runs).**

- A meets the criterion: the implementation is validated on this split.
- A does not, B does: our OS-SCL is a deviating variant; it is removed from the
  counts and the conclusions of the article.
- Neither does: the current state is kept.

**Differences that remain with the published setting (both arms):** 16 classes
against 41; 13,119 training clips against 95 % of 36,283; no ToyCar,
ToyConveyor or additional training set (non-commercial); one seed, against
published values without dispersion (seed standard deviation of OS-SCL in the
study's −6 dB cell: 0.5 point).

**Result (runs of 2026-10-05, reported on 2026-10-06).** Code identical to the
pre-registration commit `ee91ea6` (arm A run at `41b5125`, arm B and control at
`ba45534`, clean tree; these commits change only the article). Full tables per
type and per machine: `results/dcase2020_check.md`.

| Arm | Mean AUC of the 4 types | Distance to 98.53 | Criterion |
|---|---|---|---|
| A (study configuration) | 92.13 | 6.40 | not met |
| B (four repository values) | 89.12 | 9.41 | not met |
| Control (Wilkinghoff) | 98.66 | — | no criterion |

**Consequence (fixed before the runs): neither arm meets the criterion, the
current state is kept.** OS-SCL stays in the study as our reimplementation, with
its gap to the published level stated; the results of the main study do not
change. The control is not used in the article (it is not Wilkinghoff's
validation criterion).

Facts recorded with the result, not used to reinterpret it: the weakest
machines of arm A are those of the study's −6 dB cell (valve `id_02` 51.14,
fan `id_04` 75.65, pump `id_02` 79.06; the control reaches 100.00, 97.10 and
98.22 on them). In arm B the EMA model, with the repository's decay 0.9999 over
61,500 steps, classifies 52.8 % of the training clips of slider `id_02` (the
trained weights 100 %): its EMA has not fully converged, as deviation O-EMA
anticipated for this decay at this number of steps.

## Wilkinghoff (SSL / FeatEx)

Decided on 2026-09-30: model = "proposed approach" of the paper (regular loss,
mixup, StatEx variant, FeatEx), re-implemented from the author's articles
(`deviations.md`); input lengths as in `protocol.md`.

**Hypothesis on the training data (H), written before any run.** The paper does
not say on which data its models are trained. The author's DCASE 2022 report, his
ICASSP 2023 paper and his thesis (Sec. 5.3.1: classes "present in the training
datasets of the development and evaluation set", 342 classes) train one network
on the training data of the development and evaluation sets together; his DCASE
2023 report does the same (186 classes). H: for each dataset, one model is
trained on the training data of both sets and scored on both test sets. H is
inferred, not stated in the paper, and applies to the development and evaluation
cells alike (same model).

**Metrics computed for every run** (from the saved per-clip scores), per section,
then harmonic mean over the sections of the set, ×100:

- AUC and pAUC (p = 0.1) of the domain-independent ("mixed") case: all test clips
  of the section (DCASE 2022 evaluator, "AUC (all)" and "pAUC");
- per domain, **official definition**: normal clips of the domain and all
  anomalous clips of the section (DCASE 2022 evaluator; DCASE 2023 task
  description, Eq. (1)), AUC and pAUC;
- per domain, **restricted definition**: the clips of the domain only (normal and
  anomalous), AUC and pAUC.

**Pre-registered cells (criterion, fixed on 2026-09-30 before any run).** Paper,
Table 1, "proposed approach", mean ± standard deviation over five trials. The 12
cells whose metric has a published definition, each compared with our value
under that definition:

| Set | AUC source (official) | AUC target (official) | AUC mixed | pAUC mixed |
|---|---|---|---|---|
| DCASE 2022 dev | 86.0 ± 0.9 | 78.2 ± 0.7 | 82.5 ± 0.8 | 68.2 ± 1.1 |
| DCASE 2022 eval | 77.7 ± 0.8 | 71.6 ± 1.0 | 74.2 ± 0.3 | 61.2 ± 0.3 |
| DCASE 2023 dev | 71.2 ± 1.6 | 75.0 ± 1.5 | 73.1 ± 0.9 | 57.3 ± 0.6 |

All published standard deviations are ≤ 3.0 (selection rule of MobileNetV2). A
cell agrees if the mean of our five runs (five seeds) is within **3.0 points** of
the published mean.

**For information, outside the criterion:** the six per-domain pAUC cells (source
and target, three sets), and the AUC source/target cells under the restricted
definition, each reported with the definition used written out. Comparing each
published value with both definitions also indicates which one the author used.

**Excluded:** DCASE 2023 eval (ground truth under an NTT evaluation license,
non-commercial). Consequently, under H the DCASE 2023 model is still trained on
the training data of both sets (the training data are CC BY 4.0), but only its
development test set is scored.

**Data** (CC BY 4.0, downloaded for this check only, not part of the study):
DCASE 2022 Task 2 development, additional training and evaluation sets (Zenodo
6355122, 6462969, 6586456) with the evaluation ground truth of
Kota-Dohi/dcase2022_evaluator (MIT); DCASE 2023 Task 2 development and additional
training sets (Zenodo 7882613, 7830345).

**Timing run (2026-09-30, authorized; no metric computed).** One epoch on the
DCASE 2022 development training data (20,999 clips, 159 classes, 329 steps of 64)
on one RTX 5000 Ada: training about 25 s (0.076 s per step), backend (embeddings
of the training clips and k-means) about 25 s; loading the seven archives 155 s
(CPU). Under H, one DCASE 2022 model trains on about 42,000 clips for 10 epochs
(about 6,600 steps: about 8 min) and one DCASE 2023 model on about 14,000 clips of
18 s (about 2,200 steps, about 1.8 times the cost per step: about 5 min); with
the backends and scoring, about 16 min of GPU per seed and 1.3 GPU-hours for the
five seeds, plus data loading on the CPU. Data to download: 10.4 GB.

Notes from the data: the development set has 159 classes (machine type ×
section × attribute values), consistent with the 342 classes of development and
evaluation sets together reported by the author only once the additional training
data are counted (to be checked after their download); bearing section 01 has 9
target-domain training clips instead of 10 in the released archive (MD5 checked).

**Timing of OS-SCL (2026-09-30, one epoch, not a result).** Training on the main
split at 0 dB (all machine types, one model): about 40 s per epoch on one RTX
5000 Ada (0.25–0.3 s per step of 64 clips; the TFgram branch at 32,000 time
steps dominates). The internal-control run (300 epochs) therefore needs about
3.4 GPU-hours; a study run (300 epochs per training condition and seed) the
same. Determinism: three trainings of one epoch with the same seed gave
identical losses and identical scores. After one epoch the EMA weights (decay
0.9999) are still close to their initialization (ID accuracy 0.03, chance
level), as expected.

**Test of H by the class counts (2026-09-30, after the download; no run).** With
all training data of both sets (DCASE 2022: 41,999 clips; DCASE 2023: 14,000),
machine type × section × attribute values gives 323 and 167 classes, 19 fewer
than the author's 342 and 186 in both datasets. Adding the attribute values found
only in the file names of the development test clips (19 in each dataset, all
target-domain values) gives exactly 342 and 186. The training data of H are thus
consistent with the author's counts once this class list is used; the fidelity
runs use it (`deviations.md`, "Class counts").

### OS-SCL internal controls (run of 2026-09-30)

Tables: [`results/verification/osscl_tables.md`](../results/verification/osscl_tables.md)
(`scripts/controls_report.py --model osscl`). Main split, training at 0 dB, seed
0, 300 epochs (41,400 steps), 3.7 h on one GPU. **The AUCs in the tables are
implementation checks, not results of the study.**

- Training loss: 27.7 (epoch 1), 15.0 (50), 12.2 (100), 8.4 (200), 7.5 (300),
  still decreasing at the end.
- Machine-ID accuracy on the normal test clips (EMA model, 15 classes): 0.329 at
  −6 dB, 0.858 at 0 dB (training SNR), 0.912 at +6 dB.
- **Three classes are poorly identified at the training SNR:** valve `id_02` (0 of
  109 clips; predicted as pump, slider or valve `id_04`), valve `id_06` (32 of
  93), fan `id_04` (201 of 346). The other twelve are at 0.82 to 1.00. This is not
  a labelling problem: MobileNetV2 identifies the same valve clips with 100 %
  accuracy. It is a weakness of this OS-SCL model on these data. Candidates, not
  tested: the number of optimization steps (300 epochs on this training set are
  41,400 steps, about a quarter of the steps the same 300 epochs give on the 41
  machine IDs of the paper, with an EMA of decay 0.9999 whose time constant is
  10,000 steps); a defect of the reconstruction, which no external reference can
  rule out (protocol, "Known limitations").
- Determinism: two trainings of one epoch with the same seed give identical
  scores.

### OS-SCL diagnostic (run of 2026-10-01)

Tables: [`results/verification/osscl_diagnostic.md`](../results/verification/osscl_diagnostic.md)
(`python -m noisemismatch.diagnose_osscl`, `scripts/osscl_diagnostic_report.py`).
The internal-control model was retrained with the same seed (the training is
deterministic: the EMA model gives again 0.8581 on the normal test clips at 0
dB) while recording the per-class machine-ID accuracy of the trained ("raw") and
EMA weights; both sets of weights are saved every 25 epochs. Not a result of the
study.

- **The network learns every class; the EMA model is the degraded one.** Raw
  weights: 1.000 training accuracy on all 15 classes from epoch 200; normal test
  clips at 0 dB: 0.990 overall, 1.000 for valve `id_02`, valve `id_06` and fan
  `id_04`; 1.000 at +6 dB; 0.777 at −6 dB. EMA weights (decay 0.9999): training
  accuracy at chance level (one class predicted) until epoch 225, then 0.40
  (epoch 250), 0.72 (275), 0.85 (300), with valve `id_02` at 0.000, valve `id_06`
  at 0.43 and fan `id_04` at 0.59 **on the training clips** at the end.
- **Geometry of the EMA embeddings:** the weak classes form clean groups (cosine
  to their centroid 0.98; nearest-centroid accuracy 0.98 for valve `id_02`) that
  are misplaced with respect to their ArcFace centers (cosine between centroid
  and center 0.27 for valve `id_02`; 0.77 with the raw weights) and squeezed
  together (centroids of different classes at cosine 0.95 to 0.99; 0.82 to 0.85
  with the raw weights).
- **Cause:** the EMA window (10,000 steps) covers 24 % of this training (72
  epochs) against 6 % of the authors' (19 epochs): the average mixes weights that
  are far apart and has not converged to the trained model at the end
  (`deviations.md`, "EMA and number of steps").
- For information (implementation check): AUC at 0 dB of valve `id_02` with the
  final raw weights 0.36 (EMA: 0.52; MobileNetV2 controls: 0.41); it was 0.18 at
  epoch 25. Below chance for the ID-classification scores on this machine ID in
  both models; not a symptom of the EMA problem.

No correction is applied yet: it changes the pre-registered configuration and is
to be decided on training data only.

**Verification of the converted EMA decay (criterion fixed on 2026-10-01, before
the run).** After deviation O-EMA (`deviations.md`), the internal-control run is
repeated with the converted decay (0.99961 for 41,400 steps); it becomes the
internal control of OS-SCL (the outputs of the run with 0.9999 are kept in
`seed0_ema0.9999/` and `osscl_tables_ema0.9999.md`). Criterion, on the training
clips only, with no AUC involved: at the end of the training, for **every** class,
the machine-ID accuracy of the EMA weights is within 0.01 of that of the trained
weights. The formula is not adjusted whatever the outcome; if the criterion
fails, it is reported and the question is raised again.

**Result of the verification (run of 2026-10-01, 3.7 h; it is the internal
control of OS-SCL).** Tables: [`results/verification/osscl_tables.md`](../results/verification/osscl_tables.md).
Decay 0.999609 for 41,400 steps. **Criterion met:** on the training clips, the
machine-ID accuracy of the EMA weights equals that of the trained weights, 1.000
for each of the 15 classes (largest difference 0.000; threshold 0.01). Code at
launch: commit `aedd32b` (the result file records `8bcc16b`, a later
documentation-only commit).

- Machine-ID accuracy of the EMA model on the normal test clips: 0.9925 at 0 dB
  (training SNR), 1.000 at +6 dB, 0.7675 at −6 dB (with the unconverted decay:
  0.858, 0.912, 0.329).
- At the training SNR every class is at 1.00 except slider `id_00` (0.94), the
  class with 178 training clips (protocol, "Known limitations"); at −6 dB this
  class is never recognized (0.00), the others are between 0.61 and 1.00.
- Training loss: 27.7 (epoch 1), 12.2 (100), 8.4 (200), 7.5 (300), identical to
  the previous runs (same seed; the EMA does not act on the training).
- Determinism: two trainings of one epoch with the same seed give identical
  scores. Trained and EMA weights saved (`weights_all.pt`).
- The AUCs of the tables are implementation checks, not results of the study.
