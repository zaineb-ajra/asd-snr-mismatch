# Metadata: content and license

The code of this repository is under the MIT License. **The data files of this
directory are not**: they are derived from datasets and are distributed under the
terms below.

| Files | Derived from | License |
|---|---|---|
| `mimii_inventory.csv`, `mimii_pairing_content.csv`, `mimii_identity.csv`, `mimii_split.csv`, `mimii_split_sensitivity.csv`, `mimii_split_summary.csv`, `mimii_noise_check.csv`, `mimii_noise_check_cross_type.csv` | MIMII dataset (Purohit et al., DCASE Workshop 2019; Hitachi, Ltd.; Zenodo record 3384388): file names, counts, correlation scores and splits computed from its audio | **CC BY-SA 4.0** (https://creativecommons.org/licenses/by-sa/4.0/), as the MIMII dataset. Attribution: "Derived from the MIMII dataset (Purohit et al., 2019), CC BY-SA 4.0; modified: content-based clip identities, splits and checks computed by the noisemismatch authors." |
| `dcase2020_mimii_mapping.csv`, `dcase2020_other_snrs_check.csv` | MIMII (as above) and the DCASE 2020 Task 2 development set (Zenodo record 3678171, CC BY-NC-SA 4.0) | **CC BY-SA 4.0**, as the other MIMII-derived files. They contain **only file names and CRC-32 checksums**: no audio content of DCASE 2020 or MIMII is redistributed, and a checksum does not allow a file to be reconstructed. |
| `published/` | Values transcribed from the cited papers (see each file) | Facts, cited; no license claimed. |

The file lists and MD5 digests of the Zenodo archives
(`noisemismatch/data/*_zenodo_*.csv`) are part of the code (MIT).
