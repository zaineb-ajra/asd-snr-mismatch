# Deviations from the original methods

Every difference between an implementation in this repository and the original
paper or code is listed here with its justification. Sources are cited as paper
section/table or as repository file and line at a given commit.

Reference papers are listed in [`papers/README.md`](../papers/README.md).

## Common to all models

These follow from the study design (one shared data pipeline and protocol, see
[`protocol.md`](protocol.md)), not from implementation convenience.

| # | Aspect | Original methods | This repository | Justification |
|---|---|---|---|---|
| C1 | Training/test data | Each method's own benchmark (MIMII baseline: MIMII; others: DCASE 2020/2022/2023 Task 2 data or general audio for pretraining) | MIMII, common split of `protocol.md` | The study compares model families under the same data and SNR conditions. |
| C2 | Metrics | Method-specific reporting (per ID, per section, per domain; arithmetic or harmonic means) | AUC and pAUC (`max_fpr=0.1`, scikit-learn) per machine type and ID | Identical evaluation for every model. |
| C3 | Framework | Mixed (to be confirmed per repository when porting) | PyTorch for every model | Single framework, shared data loading and evaluation. |
| C4 | Normal test clips | MIMII paper, Sec. 4: "the same number of normal segments was randomly selected". MIMII baseline code (`baseline.py`, `dataset_generator`, commit `7086d21`): the *first* N normal files in sorted file-name order, N = number of anomalous files | N normal clips drawn uniformly at random with a fixed seed, per machine ID, on clip identity (`noisemismatch/data/split.py`) | Follows the paper's text and the study protocol; a draw avoids tying the test set to the first file numbers. The split is identical at every SNR in both versions. |
| C5 | Clip identity across SNRs | Files with the same name are assumed to hold the same recording at every SNR (MIMII baseline code processes each SNR folder separately) | Identity established by content; clips not identifiable at +6 and −6 dB are excluded at every SNR (15.8 % of normal, 15.4 % of anomalous clips) | File names are not reliable across SNRs in MIMII ([`data.md`](data.md)); a cross-SNR protocol needs recording identities to prevent leakage. |
| C6 | Machine IDs | All released machine IDs | valve `id_00` removed from training and test for every model, including as an auxiliary ID class | None of its anomalous clips can be identified across SNRs; all models must see the same data. |
| C7 | Test clips | Each method's own protocol | Test sets identical across SNRs and models; strict variant and 2×2 sensitivity experiment as additional evaluations | Required for ΔAUC between SNRs; see [`protocol.md`](protocol.md), Amendments 1–2. |

**Note (2026-10-06): algorithm of the one-to-one assignment.** `protocol.md`
("Acceptance", condition 1) names the Hungarian algorithm for the one-to-one
assignment that maximizes the total score. The code has always called
`scipy.optimize.linear_sum_assignment` (`noisemismatch/data/identity.py`, since
commit `37f2f73`, 2026-09-29), which SciPy documents as a modified
Jonker-Volgenant algorithm (Crouse 2016). Both solve the same linear assignment
problem, so the optimal assignment is the same; only the way of computing it
differs (if two assignments had exactly the same total score, either one could
be returned). No identity, split or result changes. `protocol.md` is not edited;
the article names the algorithm the code uses (Jonker-Volgenant). Found during
the external review of the article.

## Autoencoder (MIMII baseline)

Source: Purohit et al. 2019, Sec. 4, and `baseline.py` / `baseline.yaml` of
MIMII-hitachi/mimii_baseline (commit `7086d21`, Keras 2.1.6, TensorFlow 1.15,
librosa 0.6.0 per its `requirements.txt`). The repository has no license: the
model is re-implemented from the paper and the configuration values, no code is
copied. Implementation: `noisemismatch/models/ae.py`, `configs/models/ae.yaml`.

Reproduced as in the original: channel 0; log-mel with 64 bands, 1024-sample FFT,
hop 512, power 2, `20/2 · log10(mel + machine epsilon)`; 5 concatenated frames
(320 inputs); dense layers 64-64-8-64-64 with ReLU and a linear output; MSE; 50
epochs; batch 512; shuffled batches with the last incomplete batch kept; Keras
`validation_split=0.1` (last 10 % of the training vectors in file order, never
trained on); clip score = mean over its vectors of the mean squared error.

Framework defaults the baseline relies on without stating them, reproduced
explicitly: librosa 0.6.0 `stft(pad_mode='reflect')` and `filters.mel(norm=1)`
(= `norm='slaney'`), both checked in the librosa 0.6.0 source; Keras `Dense`
initialization (Glorot uniform weights, zero biases); Keras 2.1.6 `adam`
(learning rate 0.001, betas 0.9/0.999, epsilon 1e-7).

| # | Aspect | Original | This repository | Justification |
|---|---|---|---|---|
| A1 | Framework | Keras 2.1.6 / TensorFlow 1.15, librosa 0.6.0 | PyTorch 2.11, librosa 0.11 with the 0.6.0 defaults set explicitly | Common framework (C3). Bit-exact equality of the features with librosa 0.6.0 was not checked. |
| A2 | Adam update | Keras folds the bias corrections into the learning rate and adds epsilon to √v | PyTorch adds epsilon to √v̂ | Differs only through epsilon (1e-7); negligible. |
| A3 | Models per condition | One model per SNR folder and machine ID | One model per machine ID and training condition (single SNR or multi-condition) | Same granularity; the multi-condition model is an addition of the study, trained with the single-SNR step budget. |
| A4 | Split | Per SNR folder: first N sorted normal files in test (C4) | Study split on content-based identities (C4, C5) | Protocol. The original split is reproduced for code validation only (`noisemismatch/verification.py`). |
| A5 | Metrics | AUC | AUC and pAUC | Common evaluation (C2). |
| A6 | `validation_split` | 0.1: the last 10 % of the training vectors, in file-name order, are never trained on | 0.0: all training vectors are used (`configs/models/ae.yaml`) | The held-out block is not random: it is the last files in name order, and file order is not neutral in MIMII (pump `id_02`, +6 dB: first-N sorted test normals vs a random draw of the same size, AUC 0.496 vs 0.608). Measured effect, AUC(0.1) − AUC(0.0) paired by seed (3 seeds): fan `id_02` +6 dB −0.004 (study split), +0.001 (original split); valve `id_04` −6 dB −0.034, −0.024; pump `id_02` +6 dB +0.065 (2.2 × the between-seed standard deviation), +0.043 (3.3 ×). The four other models train on all their data, so 0.0 also keeps the comparison fair. Fidelity of the implementation is established by the validation runs with the original split and 0.1 (`results/verification/`). |

**Validation against published AUCs** (`results/verification/ae_tables.md`;
original MIMII split, `validation_split` 0.1, 3 seeds): fan `id_02` +6 dB
0.979 ± 0.010 (published 0.99), valve `id_04` −6 dB 0.534 ± 0.028 (0.50), pump
`id_02` +6 dB 0.496 ± 0.010 (0.45), including the below-chance AUC of the last
pair. The MIMII paper reports the mean of three training runs without their
dispersion. The pump residual (+0.046) was not pursued; possible causes: librosa
0.11 instead of 0.6.0 (bit-exact equality of the features not checked), Keras vs
PyTorch implementation details (A2), and the unknown dispersion of the published
mean.

## MobileNetV2 ID classifier (DCASE 2022 baseline)

Validation data: the DCASE 2022 Task 2 development set (Zenodo 6355122, 6.2 GB,
CC BY 4.0; only the 4.5 GB of the machine types needed) is downloaded for the
fidelity check of this model only ([`validation.md`](validation.md)); it is not
part of the study.

Source: Dohi et al. 2022 (DCASE 2022 Task 2 description) and
Kota-Dohi/dcase2022_task2_baseline_mobile_net_v2 (commit `aaf3932`, MIT License;
`baseline.yaml`, `keras_model.py`, `00_train.py`, `01_test.py`, `common.py`;
Keras 2.8.0, TensorFlow 2.8.0, librosa 0.9.1 per its `requirements.txt`). The
network is `keras.applications.MobileNetV2` (Keras 2.8.0, Apache License 2.0),
rebuilt layer by layer. Implementation: `noisemismatch/models/mobilenetv2.py`,
`configs/models/mobilenetv2.yaml`.

Reproduced as in the original: channel 0 (the DCASE files are mono); log-mel with
128 bands, 1024-sample FFT, hop 512, power 2, `20/2 · log10(max(mel, machine
epsilon))`; images of 64 consecutive frames × 128 bands, replicated on 3 channels
(`Concatenate()([h, h, h])`); training on one window out of 8
(`vectors[::n_hop_frames]` in `00_train.py`), scoring on every window
(`common.file_to_vectors` in `01_test.py`); MobileNetV2 with `alpha=0.5`, no
pretrained weights and its classification top (global average pooling, dense
softmax layer); one model per machine type classifying the sections (here the
machine IDs) of that type; categorical cross-entropy, Adam with learning rate
1e-5, 20 epochs, batch 32, shuffled; training images ordered by section, then
file name (`sorted(glob.glob(...))`), which fixes which images Keras'
`validation_split` holds out; anomaly score of a clip = mean over its windows of
`log(max(1 − p, eps)) − log(max(p, eps))`, `p` the softmax probability of the
clip's own section.

Framework details reproduced explicitly: librosa 0.9.1 `stft(pad_mode='constant')`
and `filters.mel(norm='slaney')`; Keras MobileNetV2 channel rounding
(`_make_divisible`), TensorFlow "same" padding of the stride-2 convolutions (one
extra row and column at the end), batch normalization epsilon 1e-3 and moving
average momentum 0.999 (PyTorch `momentum=1e-3`), ReLU6, Glorot-uniform kernels
and zero biases; tf.keras `Adam` epsilon 1e-7. The parameter count equals Keras'
(1,987,224 for 1,000 classes; checked in `tests/test_mobilenetv2.py`).

| # | Aspect | Original | This repository | Justification |
|---|---|---|---|---|
| M1 | Framework | Keras 2.8 / TensorFlow 2.8, librosa 0.9.1 | PyTorch 2.11, librosa 0.11 with the 0.9.1 defaults set explicitly | Common framework (C3). Bit-exact equality of the features with librosa 0.9.1 was not checked. |
| M2 | Optimizer and batch normalization | Keras Adam (epsilon added to √v, bias corrections folded into the learning rate); Keras moving statistics | PyTorch Adam (epsilon added to √v̂); PyTorch running statistics with the same momentum and epsilon | As A2: differs only through epsilon. Batch normalization is equivalent: every layer of `keras.applications.MobileNetV2` sets `epsilon=1e-3, momentum=0.999` (Keras 2.8.0 source); Keras updates `moving = momentum · moving + (1 − momentum) · batch`, PyTorch `running = (1 − momentum) · running + momentum · batch`, hence PyTorch `momentum=1e-3`; both update the moving variance with the unbiased batch variance (TensorFlow 2.8.0 `fused_batch_norm_op.cc`, CPU kernel; the GPU path calls cuDNN, not inspected) and normalize with the biased one. Checked on the 52 layers of the network (`tests/test_mobilenetv2.py`). |
| M3 | Classes | 3 sections per machine type (DCASE 2022) | The machine IDs of a machine type: 4, or 3 for valve (C6) | MIMII has machine IDs, not sections; both designate the clips of one machine within a machine type, which is what the auxiliary task classifies. |
| M4 | Models per condition | One model per machine type | One model per machine type and training condition (single SNR or multi-condition) | As A3; the multi-condition model gets the step budget of a single-SNR model. |
| M5 | `validation_split` | 0.1: the last 10 % of the training images, never trained on | 0.0 in the study (`configs/models/mobilenetv2.yaml`); 0.1 in the fidelity check | As A6. Here the held-out block is also a single class: the images are ordered by section, so the last 10 % belong to the last section or, in the study, to the last machine ID of the type (about 40 % of its images when the four IDs have similar sizes). In DCASE 2022 it is the last 300 of the 1,000 training clips of section 02 in file-name order, i.e. 290 source-domain clips and all 10 target-domain clips (`source` sorts before `target`): the original baseline never trains on the target domain of section 02. Measured on ToyCar (control run, [`validation.md`](validation.md)): section-02 AUC (target) 44.42 ± 8.92 with 0.1 (5 seeds) vs 85.16 with 0.0 (seed 0). |
| M6 | Decision threshold | Gamma distribution fitted to the training scores, threshold at its 0.9 quantile, binary decisions | Not computed | Only scores are evaluated (AUC, pAUC). The gamma fit of `00_train.py` places `np.log(...)` around the difference of the two log terms (a misplaced parenthesis), which affects the decisions only. |
| M7 | Metrics | AUC per domain (source, target) and pAUC per section, harmonic means | Study: AUC and pAUC per machine type and ID (C2). Fidelity check: the per-section metrics of `01_test.py` | Common evaluation; the original metrics are kept where the published values are compared. |

## Angular margin with self-supervision (Wilkinghoff, SSL/FeatEx)

**Rule (decided on 2026-09-30).** The model is re-implemented from the author's
articles only, no code of wilkinghoff/ssl4asd (GPL-3.0) being copied: no file,
structure of files or functions, or line of code. Model: the "proposed approach"
of the paper (regular loss, mixup, StatEx variant, FeatEx). Sources, in this
order: **P** = the paper (Wilkinghoff, ICASSP 2024, arXiv 2312.09578); **A** = the
author's other articles: **T** thesis (Bonn, 2024, `wilkinghoff.com/publications/diss.pdf`),
**D23** ICASSP 2023 "Design choices…", **S21** IJCNN 2021 "Sub-cluster AdaCos",
**J24** TASLP 2024 (arXiv 2309.15643), **R22/R23** DCASE 2022/2023 technical
reports; **C** = a paper they cite (ResNet, He et al. 2016); **F** = default of
the framework the author states (TensorFlow); **R** = the repository, as a fact
(value or layer specification), each item listed in the last table below.

| Element | Specification | Source |
|---|---|---|
| Input | waveform, 16 kHz; repeated (and randomly cropped) to the longest clip of the dataset: 160,000 samples (DCASE 2022), 288,000 (DCASE 2023); MIMII study: 160,000 | R23 Sec. 2.1; D23 Fig. 1; P Fig. 1 |
| Spectrogram branch input | magnitude STFT, Hann window 1024, hop 512, no mel, no log; frames without padding (311 × 513 for 10 s, 561 × 513 for 18 s); temporal mean normalization (TMN) per sample | T Sec. 5.3.1, Table 15; D23 Sec. 2.1 |
| Spectrum branch input | magnitude DFT of the whole signal, first N/2 bins (80,000 for 10 s) | D23 Fig. 1, Table 2; T Table 14 |
| Spectrogram sub-network | 7×7 convolution, stride 2, 16 channels; four stages of two residual blocks (two 3×3 convolutions each) with 16, 32, 64, 128 channels; output sizes 156×257, 77×128, 39×64, 20×32, 10×16; max pooling over the whole time axis; flatten (2,048); dense 128 without activation (embedding) | T Table 15; D23 Table 1 |
| Residual stages | stem pooling 3×3, stride 2, and stride-2 first convolution of stages 2–4 with a 1×1 projection shortcut, as in the cited ResNet; consistent with every output size of the tables | C (He et al. 2016); sizes in T Table 15 |
| Spectrum sub-network | 1D convolutions: 128 channels, kernel 256 stride 64; kernel 64 stride **see note**; kernel 16 stride 4; "same" padding (1,250 outputs for 80,000); flatten; four dense layers of 128; dense 128 without activation (embedding) | T Table 14; D23 Table 2 |
| Activations, normalization | batch normalization and ReLU after every convolutional and dense layer except the embedding layers; no bias anywhere | T Sec. 5.3.1; D23 Sec. 2.2 |
| Embedding | concatenation of both embeddings (256) | P; T |
| Loss | two sub-cluster AdaCos losses, 16 sub-clusters per class, equally weighted: (1) N meta-information classes, mixed-up labels, non-trainable randomly initialized centers; (2) 9·N classes of the sequentially applied StatEx variant and FeatEx, trainable centers | P Sec. 3.4; T Sec. 5.7.2 |
| Sub-cluster AdaCos | Definition 4 (adaptive scale with f_max and B̂avg over all logits of the mini-batch); with mixup, θ̂med = median of the mixed-up angles λθ_i + (1−λ)θ_j | J24 Def. 4; S21 Eqs. (4)–(9) |
| Classes | all combinations of machine type, section and attribute values in the training data of the development and evaluation sets (342 for DCASE 2022, 186 for DCASE 2023); MIMII study: machine type × machine ID | T Sec. 5.3.1; D23; R23; see the note on class counts |
| Mixup | on waveforms, λ ~ U(0, 1), probability 50 % per sample (100 % without SSL) | P Sec. 3.1; R23 Sec. 3.1; T Sec. 5.7.2 |
| StatEx variant | temporal StatEx (statistics over the frequency axis), applied to the spectrogram features, probability 50 %, labels (0, 0.5·y1, 0.5·y2) | P Sec. 3.2; T Def. 5.5 |
| FeatEx | exchange of the spectrogram-branch embedding of x1 and the spectrum-branch embedding of x2, probability 50 %, labels (0, 0.5·y1, 0.5·y2) | P Sec. 3.3; T Def. 5.6 |
| Training | 10 epochs, batch 64, Adam | P; D23; T |
| Adam, batch norm, initialization | TensorFlow 2 defaults: learning rate 1e-3, β = (0.9, 0.999), ε = 1e-7; batch norm momentum 0.99, ε = 1e-3; Glorot-uniform kernels | F |
| Backend | per section (DCASE 2022) or machine type (DCASE 2023): k-means with 16 clusters on the embeddings of the source-domain normal training clips; cosine distance to the means and to the target-domain training embeddings; score = minimum; MIMII study: k-means on the training embeddings of the machine ID (no target domain) | P Sec. 2; T Sec. 5.3.1; J24 Sec. IV-B |

**Stride of the second 1D convolution: 32 (decided on 2026-09-30).** D23 Table 2
and T Table 14 both write "64, stride = 16" for the second convolution, but give
its output size as 40 × 128 and the flattened size as 1,280. With "same" padding
(which the first size already requires: 80,000 / 64 = 1,250), the sizes follow
from the strides: stride 32 gives ⌈1,250 / 32⌉ = 40, then ⌈40 / 4⌉ = 10 after the
third convolution, and 10 × 128 = 1,280, exactly the two written sizes; stride 16
gives ⌈1,250 / 16⌉ = 79, then 20, and a flattened size of 2,560, matching neither.
Two written values that can be checked against each other outweigh one isolated
value, most likely a typo repeated from the article into the thesis. The
repository also uses 32 (a fact, not the basis of the decision).

**Class counts (deduced, 2026-09-30).** Counting the combinations of machine
type, section and attribute values in the training data of both sets gives 323
classes for DCASE 2022 and 167 for DCASE 2023, 19 fewer than the author's 342 and
186 in both datasets, although the two datasets have different numbers of
machine types (7, 14) and sections (42, 14). The gap is not a domain split
(adding the domain to the class adds no class) and no training file lacks
attributes. It is closed exactly by also counting the attribute values that
appear only in the file names of the development test clips: 19 in each dataset,
all of them target-domain values (DCASE 2022: ToyCar 00, ToyTrain 00, bearing 00
and 01, gearbox 00, slider 00 and 01; DCASE 2023: bearing, gearbox, slider).
The fidelity runs therefore use this deduced class list: the 19 classes have no
training clip and only add class centers to the losses (they enter the softmax
denominators and the adaptive scale). Only attribute values in file names are
used, never test labels or scores. The regular count of 19 in both datasets is a
coincidence of this definition, reported as an observation (also of possible
interest to the author).

**Interpretations** (the text leaves a choice; ours, not taken from the
repository): one forward pass per batch, both losses computed on it (standard
loss on the concatenated embedding with the mixed-up labels, SSL loss on the
FeatEx output with the 9·N labels); for sub-clusters, the angle of a sample to a
class is the angle to the closest sub-cluster of that class; mixup, StatEx and
FeatEx partners are drawn by a random permutation of the mini-batch.

**Present in the repository but in none of the articles, hence not implemented**
(consequence of the paper-only rule; a possible source of disagreement with the
published numbers): squeeze-and-excitation blocks in every residual block and
after every 1D convolution; L2 weight decay (1e-5) on every layer.

**Values and layer specifications taken from the repository** (each one listed;
no code derived):

| Item | Value | Location in wilkinghoff/ssl4asd (commit `37aa21d`) |
|---|---|---|
| Embeddings before k-means | L2-normalized (length normalization) before k-means and before the cosine distances | `main_statex+featex.py`, lines 507–510 and 514 |

This agrees with our choice made before the look-up, which also follows from the
loss: sub-cluster AdaCos works on the unit hypersphere (cosine similarities), so
the clustering is done where the embeddings are trained to be compared.

**Our choices for values not found in the articles** (kept, decided on
2026-09-30): k-means with scikit-learn defaults (`n_init="auto"`, k-means++
initialization, seed = run seed); ε = 1e-12 added to the standard deviation in
StatEx; for DCASE 2023, the repetition and random crop to 18 s are done once per
clip when the data are loaded, with a generator seeded by the CRC-32 of the file
name (the same crop in every run).

**Data note (documented, not corrected).** In the DCASE 2022 development set as
released (MD5 checked), bearing section 01 has 9 target-domain training clips
instead of 10.

Deviations: to be completed with the implementation.

## OS-SCL (Huang et al.)

**Rule (decided on 2026-09-30).** Re-implemented from the paper (Huang et al.,
ICASSP 2025, arXiv 2509.13853) and the papers it cites; no code of huangswt/OS-SCL
is copied or ported, not even the parts it declares under a permissive license
(`model/waveq.py`): its own code declares derivations from repositories without a
license (STgram-MFN, arcface-pytorch) and under CC BY-NC 4.0 (mixup-cifar10).
Variant: TFSTgram (OS-SCL).

**Order of sources for a value the paper does not give** (anomaly score, EMA
decay, weight decay, MobileFaceNet configuration, and any other): (1) the OS-SCL
paper; (2) the paper it cites for that component (for the anomaly score:
STgram-MFN, Liu et al., ICASSP 2022, −log p(ID)); (3) the huangswt/OS-SCL
repository, as a documented fact, no code derived. The level used is stated for
each value in the table below.

Levels: **1** OS-SCL paper; **2** a paper it cites: STgram-MFN (Liu et al.,
ICASSP 2022), MobileFaceNets (Chen et al., 2018), Noisy-ArcMix (Choi et al.,
ICASSP 2024), mixup, ArcFace; **3** fact of the repository huangswt/OS-SCL at
commit `9aca394` (value or layer specification, no code). Implementation:
`noisemismatch/models/osscl.py`, `configs/models/osscl.yaml`.

| Element | Specification | Level | Source |
|---|---|---|---|
| Input features | Concat(log-Mel, Tgram, TFgram): 3 × 128 × 313 | 1 | Eq. (3), Sec. III-A |
| Log-Mel | 128 bins, window 1024, hop 512 (313 frames for 10 s, centered frames), power 2, natural log | 2 | STgram-MFN Eq. (1), Sec. 3.1 |
| Log offset | 1e-8 | — | not given anywhere (our choice) |
| Tgram | Conv1D(1→128, kernel 1024, stride 512, padding 512), then 3 × [layer norm, LeakyReLU, Conv1D(128→128, kernel 3)] | 2 | STgram-MFN Table 1 |
| Tgram details | layer norm over the time axis (313); LeakyReLU slope 0.2; no bias | 3 | `model/net.py` (TgramNet) |
| TFgram | Conv1D(64, k 11, s 5, p 5), BN, ReLU; block(64); MaxPool1D(4); block(64); adaptive max pool 626; block(128); adaptive max pool 313; block = 2 conv (k 3; p 1, then p 2 dilation 2) with BN and ReLU | 1 | Table I, Sec. II-D |
| Embedding network | MobileFaceNet as in its paper: conv 3×3 s2 64; depthwise 3×3 64; bottlenecks (t, c, n, s) = (2,64,5,2), (4,128,1,2), (2,128,6,1), (4,128,1,2), (2,128,2,1); conv 1×1 512; linear global depthwise conv (8 × 20 for this input); linear conv 1×1 128; PReLU; BN | 2 | MobileFaceNets Table 1, Sec. 3.3 |
| Feature perturbation head | Linear 128→64, ReLU, Linear 64→128 (TFSTgram), L2 normalization; used for both losses | 1 | Sec. II-A, II-B, III-A |
| Head details | batch norm after each ReLU | 3 | `model/net.py` (BasicHead) |
| Loss | noise-supervised contrastive loss (positives by original label y_a, τ = 0.02) + Noisy-ArcMix (ArcFace s = 30, m = 0.4 on y_a, target λ y_a + (1−λ) y_b) | 1, 2 | Eqs. (1), (2), (4); Noisy-ArcMix Eq. (4) |
| Mixup | waveform and log-Mel mixed with the same λ ~ Beta(0.5, 0.5) and the same in-batch permutation; one λ per batch | 1, 2 | Sec. II-A, III-A; mixup |
| Training | one model for all machine types; batch 64; 300 epochs; AdamW, learning rate 1e-4, cosine annealing; EMA of the weights | 1 | Sec. III-A |
| AdamW weight decay | 0.01 (PyTorch default, not overridden) | 3 | `trainer.py` |
| Cosine annealing | T_max = number of epochs, stepped per epoch, minimum 0.1 × learning rate | 3 | `trainer.py` |
| EMA | repository: decay 0.9999 per step; **here: decay = 1 − 1 / (0.061843 × number of steps of the run)** (deviation O-EMA below); the EMA step also multiplies the trained weights by (1 − 2e-6); evaluation with the EMA model | 3, adapted | `losses.py` (WeightEMA), `trainer.py`, `eval.py` |
| Anomaly score | −log p(machine ID of the clip), softmax of the ArcFace logits without margin | 2 | STgram-MFN Sec. 3.1 ("negative log probability") |
| Training set of the authors | development and additional training data of DCASE 2020 Task 2: 36,283 clips of 41 machine IDs (counted in the archive directories of Zenodo 3678171 and 3727685, file names only); 95 % used for training (random split), batch 64, last batch kept: 539 steps per epoch, 161,700 steps in 300 epochs | 1, 3 | Sec. III-A; `train.py`, `utils.py` (`dataset_split`), `datadownload.sh` |

**Deviation O-EMA: the EMA decay is converted to the number of steps of the run
(decided on 2026-10-01, before any test result of the corrected model).**

- *Published value.* The paper says only that EMA smooths the parameters; the
  repository uses a decay of 0.9999 per optimization step, i.e. an averaging
  window of 1 / (1 − 0.9999) = 10,000 steps.
- *What the value means for the authors.* Their training set has 36,283 clips of
  41 machine IDs, 95 % of which are trained on with batches of 64: 539 steps per
  epoch, **161,700 steps** in 300 epochs. The window is 10,000 / 161,700 = **6.2 %
  of the training** (18.6 epochs).
- *What it means here.* On the main split, 300 epochs are **41,400 steps** (138
  per epoch; the multi-condition runs have the same step budget). The same
  10,000-step window would be **24.2 % of the training** (72.5 epochs).
- *Observed consequence with 0.9999* (diagnostic of the internal-control model,
  [`validation.md`](validation.md)): the trained weights reach a machine-ID
  accuracy of 1.000 on the training clips for every class, but the EMA model
  stays at chance level until epoch 225 and ends at 0.853, with one class at
  0.000 on its own training clips. The EMA has not converged to the trained
  model, and the EMA model is the one that is evaluated.
- *Rule, applied mechanically to every run and never adjusted afterwards.* The
  window keeps the authors' fraction of the training:
  `decay = 1 − 1 / (ema_window_fraction × steps)`, with
  `ema_window_fraction = 10,000 / 161,700 = 0.061843` (`configs/models/osscl.yaml`)
  and `steps` the number of optimization steps of the run. Main split: window
  2,560 steps, **decay 0.99961**. Sensitivity split: the same formula with its own
  number of steps. With the authors' 161,700 steps the formula returns 0.9999.
- *Basis of the decision.* The conversion follows from the numbers of steps
  alone. It was decided on the training-data evidence above (ID accuracy on
  training clips), before any run with the converted decay and without using
  any test AUC. Verification criterion, on training clips only, fixed before the
  verification run: [`validation.md`](validation.md).
- *Safeguard in every run.* The machine-ID accuracy of the trained and of the
  EMA weights on the training clips is recorded at the end of every OS-SCL run
  (`model_diagnostics` in `result.json`).
- *Not changed.* Two other per-step constants of the repository keep their
  values: the multiplication of the trained weights by (1 − 2e-6) at each EMA
  step (cumulated over the training: 0.72 for the authors, 0.92 here) and the
  AdamW weight decay. Neither prevents the trained weights from reaching 1.000
  on the training clips; they are reported, not converted.

**Interpretations** (ours): the contrastive loss is averaged over the anchors that
have a positive (Eq. (1) writes a sum); ArcFace uses cos(θ + m) as written, without
the fallback used in some implementations.

**Differences between the repository and the cited papers, resolved by the order
of sources** (cited paper first): log-Mel window 1024 and natural log (repository:
2048 and decibels); MobileFaceNet with the five bottleneck sequences of its paper
(repository: three, (2,128,2,2), (4,128,2,2), (4,128,2,2)); anomaly score without
margin (repository: cross-entropy with the margin). Parameter count, for
information: 1.49 M with the MobileFaceNets configuration, 1.30 M with the
repository's; the OS-SCL paper reports 1.38 M (Table V), which decides neither.

**OS-SCL, arm B of the DCASE 2020 diagnostic (2026-10-05, after the study
results; [`validation.md`](validation.md)).** The four values of the repository
(commit `9aca394`) recorded as facts on 2026-09-30 in the paragraph above, used
only by arm B; no code of the repository is used. The repository was not
re-read on 2026-10-05.

| Value | Study (and arm A) | Repository fact | Implementation in arm B |
|---|---|---|---|
| Log-Mel spectrogram | window 1024, natural log, offset 1e-8 | window 2048, decibels | `logmel_n_fft: 2048`; `log_scale: db` = 10 log10(max(power, 1e-10)) (torchaudio `AmplitudeToDB`, power, no `top_db`). The record of 2026-09-30 says only "decibels": the floor is ours. |
| MobileFaceNet | five bottleneck sequences (its paper) | three sequences: (2,128,2,2), (4,128,2,2), (4,128,2,2) | `mobilefacenet: repository`; the other layers as in the study (stem, 1×1 to 512, global depthwise 8 × 20, linear to 128): 1.30 M parameters |
| Anomaly score | −log p(machine ID), no margin | cross-entropy with the margin | `score_margin: true`: −log softmax(s cos(θ + m·1[ID]))[ID], s = 30, m = 0.4 |
| EMA decay | 1 − 1 / (0.061843 × steps) (O-EMA) | 0.9999 | `ema_decay: 0.9999` |

Not covered by the record, so unchanged in arm B: the Tgram kernel (1024, from
STgram-MFN Table 1; whether the repository ties it to the 2048 window was not
recorded) and the layers of MobileFaceNet other than the bottleneck sequences.

Deviations: to be completed with the study runs (inputs: MIMII channel 0, 10 s;
classes: machine type × machine ID).

## EAT frozen encoder + kNN (Chen et al.)

Source: Chen et al., "EAT: Self-Supervised Pre-Training with Efficient Audio
Transformer", IJCAI 2024; checkpoint and inference code of the Hugging Face
repository `worstchan/EAT-base_epoch30_pretrain` at revision `6eb9593` (MIT
License, Copyright (c) 2024 Wenxi Chen; code from cwx-worst-one/EAT), weights
SHA-256 `8623072d…`. EAT is a general-purpose audio encoder, not an ASD method:
there is no original ASD configuration to reproduce. The ASD use (frozen
embeddings, kNN) and its configuration are choices of this study, fixed before
any embedding was computed ([`protocol.md`](protocol.md), "EAT + kNN
configuration"); the embedding and scoring follow Saengthong & Shinozaki
(arXiv 2603.13749, 2026): temporal mean of the last-layer patch tokens, K = 1.
Implementation: `noisemismatch/features/eat_encoder.py` (vendored encoder),
`noisemismatch/features/eat.py` (preprocessing, pooling),
`noisemismatch/models/eat_knn.py`, `configs/models/eat_knn.yaml`.

Reproduced as in the model card: mean removal; Kaldi filterbank (128 bins, 25 ms
Hanning window, 10 ms shift, HTK-compatible, no dither); zero padding to 1,024
frames; normalization (x + 4.268) / (2 × 4.569); encoder output of
`extract_features` (CLS + patch tokens of the last layer).

| # | Aspect | Original | This repository | Justification |
|---|---|---|---|---|
| E1 | Code | `transformers` wrapper with `trust_remote_code`, `timm` helpers | Vendored inference code at the pinned revision, without `transformers` or `timm` (`Mlp`, attention and blocks copied; dropout, drop-path and unused branches removed; positional table loaded from the checkpoint); checkpoint loaded with `strict=True` | No remote code execution, which would also break reproducibility if the upstream repository changed. Equivalence checked against the official path ([`validation.md`](validation.md), "EAT + kNN", check (a)). |
| E2 | Numerics | PyTorch defaults (TF32 allowed for cuDNN convolutions) | Float32 with TF32 disabled for matrix products and convolutions | Full float32 precision; affects rounding only. |
| E3 | Batching | One clip per forward pass (model card) | Batches of 32 clips | Speed; rounding only (covered by check (a)). |
| E4 | Input | Any audio; resampled to 16 kHz | Channel 0 of MIMII (16 kHz, 10 s: 998 frames, zero-padded to 1,024) | Common to all models (channel 0). |
| E5 | ASD scoring | None (general encoder) | Cosine distance to the nearest training embedding of the same machine ID (K = 1), temporal-mean pooling; multi-condition: reference set of the three SNRs | Design of this study (protocol); secondary variants (CLS, K = 5) reported separately and never used in place of it. |
| E6 | Seeds | — | No random choice: the three seeds give identical results | Stated in the result tables (protocol). |
