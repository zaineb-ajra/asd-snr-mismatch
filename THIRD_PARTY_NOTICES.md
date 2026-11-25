# Third-party notices

This repository is released under the MIT License ([`LICENSE`](LICENSE)). It
contains code ported or vendored from the projects below, each under a
permissive license; the license terms travel with that code. The list of
components, with what is used and where, is in the README ("Third-party code and
licenses").

## Datasets (not redistributed)

- **MIMII dataset** — Purohit, H., R. Tanabe, K. Ichige, T. Endo, Y. Nikaido,
  K. Suefusa and Y. Kawaguchi (2019). MIMII Dataset: Sound Dataset for
  Malfunctioning Industrial Machine Investigation and Inspection. In Proceedings
  of the Detection and Classification of Acoustic Scenes and Events 2019
  Workshop (DCASE2019), pp. 209–213. Hitachi, Ltd.; Zenodo record 3384388.
  License: CC BY-SA 4.0. No audio is redistributed. The files of `metadata/`
  derived from it (inventory, content-based identities, splits, checks) are
  distributed under CC BY-SA 4.0 ([`metadata/README.md`](metadata/README.md)).
- **DCASE 2020 Challenge Task 2 development set** — Koizumi, Y., Y. Kawaguchi,
  K. Imoto, T. Nakamura, Y. Nikaido, R. Tanabe, H. Purohit, K. Suefusa, T. Endo,
  M. Yasuda and N. Harada (2020). Description and Discussion on DCASE2020
  Challenge Task2: Unsupervised Anomalous Sound Detection for Machine Condition
  Monitoring. In Proceedings of the Detection and Classification of Acoustic
  Scenes and Events 2020 Workshop (DCASE2020), pp. 81–85. Zenodo record 3678171.
  License: CC BY-NC-SA 4.0. No audio of this set is downloaded or redistributed.
  `metadata/dcase2020_mimii_mapping.csv` and
  `metadata/dcase2020_other_snrs_check.csv` are derived from the file names and
  CRC-32 checksums listed in the central directories of its ZIP archives (read by
  HTTP range requests), matched to MIMII files; they hold no audio content, and a
  checksum does not allow a file to be reconstructed.

## MIT License components

The following copyright notices apply to the files named after them, under the
MIT License terms reproduced at the end of this section.

- **DCASE 2022 Task 2 MobileNetV2 baseline** —
  https://github.com/Kota-Dohi/dcase2022_task2_baseline_mobile_net_v2, commit
  `aaf3932`. Copyright (c) 2022 Kota-Dohi. It derives from the DCASE 2020 and
  2021 baselines (https://github.com/y-kawagu/dcase2020_task2_baseline,
  https://github.com/y-kawagu/dcase2021_task2_baseline_mobile_net_v2), Copyright
  (c) 2020 and (c) 2021 Hitachi, Ltd., MIT License. Ported to PyTorch in
  `noisemismatch/models/mobilenetv2.py` and
  `noisemismatch/verification_dcase2022.py`.
- **EAT** — https://github.com/cwx-worst-one/EAT and the Hugging Face repository
  `worstchan/EAT-base_epoch30_pretrain`, revision
  `6eb95936bf1061cacee1c0b89404a1aca9a3c62a`. Copyright (c) 2024 Wenxi Chen.
  Vendored in part in `noisemismatch/features/eat_encoder.py`; model-card
  preprocessing in `noisemismatch/features/eat.py` and `scripts/eat_reference.py`.
- **data2vec 2.0 (fairseq)** — https://github.com/facebookresearch/fairseq,
  `examples/data2vec/models/modalities/modules.py`, through the EAT code.
  Copyright (c) Facebook, Inc. and its affiliates. Attention and block modules in
  `noisemismatch/features/eat_encoder.py`.

MIT License terms:

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

## Apache License 2.0 component

- **Keras MobileNetV2** — `keras/applications/mobilenet_v2.py`, Keras 2.8.0
  (https://github.com/keras-team/keras). Copyright 2018 The TensorFlow Authors.
  All Rights Reserved. Licensed under the Apache License, Version 2.0
  ([`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt)). Translated to PyTorch
  in `noisemismatch/models/mobilenetv2.py` (network definition, channel rounding,
  "same" padding); changes are stated in that file's header.
