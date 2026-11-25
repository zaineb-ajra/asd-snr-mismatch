# Source papers

Reference papers for the methods compared in this study. Used to reproduce
architectures, features and hyperparameters as faithfully as possible.
Any deviation from these papers is documented in `docs/deviations.md`.

| File | Paper | arXiv version | Role in this repo | Official code |
|---|---|---|---|---|
| purohit2019_mimii.pdf | Purohit et al., "MIMII Dataset", DCASE Workshop 2019 (arXiv:1909.09347) | v1 | Dataset + autoencoder baseline | github.com/MIMII-hitachi/mimii_baseline |
| dohi2022_dcase2022_task2.pdf | Dohi et al., "Description and Discussion on DCASE 2022 Challenge Task 2", DCASE 2022 (arXiv:2206.05876) | v2 | MobileNetV2 ID-classifier baseline | github.com/Kota-Dohi/dcase2022_task2_baseline_mobile_net_v2 |
| wilkinghoff2024_ssl4asd.pdf | Wilkinghoff, "Self-Supervised Learning for Anomalous Sound Detection", ICASSP 2024 (arXiv:2312.09578) | v1 | Angular-margin / FeatEx model | github.com/wilkinghoff/ssl4asd |
| huang2025_osscl.pdf | Huang et al., "Noise Supervised Contrastive Learning and Feature-Perturbed for Anomalous Sound Detection", ICASSP 2025 (arXiv:2509.13853) | v2 | OS-SCL contrastive model | github.com/huangswt/OS-SCL |
| chen2024_eat.pdf | Chen et al., "EAT: Self-Supervised Pre-Training with Efficient Audio Transformer", IJCAI 2024 (arXiv:2401.03497) | v1 | Frozen pretrained encoder + kNN | github.com/cwx-worst-one/EAT |

These PDFs are arXiv preprints. They are for local reference only and are not
redistributed with the repository (see .gitignore). To fetch them:

```bash
cd papers
curl -L -o purohit2019_mimii.pdf        https://arxiv.org/pdf/1909.09347v1
curl -L -o dohi2022_dcase2022_task2.pdf https://arxiv.org/pdf/2206.05876v2
curl -L -o wilkinghoff2024_ssl4asd.pdf  https://arxiv.org/pdf/2312.09578v1
curl -L -o huang2025_osscl.pdf          https://arxiv.org/pdf/2509.13853v2
curl -L -o chen2024_eat.pdf             https://arxiv.org/pdf/2401.03497v1
```
