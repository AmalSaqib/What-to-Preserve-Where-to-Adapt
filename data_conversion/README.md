# Data conversion

Scripts that turn the public downloads of the three datasets into the nnU-Net
v1 task folders used in the paper, and a verifier that checks the result
against fingerprints of the exact data we trained and evaluated on.

| Dataset | Public source | Script | Output | Reproduced |
|---|---|---|---|---|
| UMD | [figshare 10.6084/m9.figshare.23541312.v3](https://doi.org/10.6084/m9.figshare.23541312.v3), `UMD.zip` (MD5 `f52985246c363a7fb9c8f6ae33dcde84`) | `convert_umd.py` | `Task666_UMD` | All 798 files byte-identical |
| ECPC-IDS | [figshare 10.6084/m9.figshare.23808258.v1](https://doi.org/10.6084/m9.figshare.23808258.v1), `ECPC-IDS.zip` (MD5 `9cac6378d62bef4d35fbc1ed2ed5cbd9`) | `convert_ecpc.py` | `Task555_ECPC` | Near-exact; see below |
| UT-EndoMRI (T1FS) | [Zenodo 10.5281/zenodo.15750762](https://doi.org/10.5281/zenodo.15750762), `UT-EndoMRI.zip` (MD5 `7ace6e1b08efa10d0a1967073b0ba41c`) | `convert_ut_endomri.py` | `Task778_EndoMRI_T1FS` | All 309 files byte-identical |

Requires `nibabel`, `numpy` and `SimpleITK` (all in the repository
environment).

## Usage

```bash
# 1. Download and extract the archives listed above (check the MD5 sums).
# 2. Convert into the nnU-Net raw-data folder:
R=$nnUNet_raw_data_base/nnUNet_raw_data
python data_conversion/convert_umd.py        downloads/UMD          $R/Task666_UMD
python data_conversion/convert_ecpc.py       downloads/ECPC-IDS     $R/Task555_ECPC
python data_conversion/convert_ut_endomri.py downloads/UT-EndoMRI   $R/Task778_EndoMRI_T1FS
# 3. Verify against the reference fingerprints:
python data_conversion/verify.py $R/Task666_UMD          data_conversion/reference/Task666_UMD.csv
python data_conversion/verify.py $R/Task555_ECPC         data_conversion/reference/Task555_ECPC.csv
python data_conversion/verify.py $R/Task778_EndoMRI_T1FS data_conversion/reference/Task778_EndoMRI_T1FS.csv
```

Then continue with `reproducibility/preprocess.py` as described in the main
README.

## What the scripts do

All three tasks share one backbone with two input channels: the anatomical
image (MRI or CT) first and PET second. For the MRI datasets the second
channel is all zeros.

- **UMD** copies each `UMD_221129_NNN` T2 image and segmentation unchanged as
  `UMD_NNN`. 34 public cases are unused because their segmentation file is not
  a readable NIfTI (33 are named `*_seq.nii.gz`, one `UMD_2221129_037_seg`).
- **ECPC-IDS** reads the 3D DICOM part (`subset_A`), keeps the PET slices that
  lie inside both the CT and the label range, resamples CT linearly onto the
  PET grid, and stores CT (int32 HU), PET (float64) and the tumor mask
  (uint8). 134 of the 155 public cases are used.
- **UT-EndoMRI** copies each T1FS image as `EndoT1FS_{site}{patient}` and
  merges the per-structure label files (up to three raters at site 1) by
  majority vote on the image grid, breaking ties as endometrioma > cyst >
  ovary > uterus. Labels: 1 uterus, 2 ovary, 3 endometrioma, 4 cyst.

The training/held-out splits were drawn at random once (UT-EndoMRI: seed-42
shuffle, see `reference/Task778_EndoMRI_T1FS.fixed_split_80_20.json`). The
scripts read the case selection and split from the reference manifests, so
they reproduce them exactly.

## Reference fingerprints

`reference/*.csv` record, for every file of the original task folders, its
split, SHA-256, a hash of the voxel array, shape, spacing, affine, data type
and label values. `make_manifest.py` creates such a manifest; `verify.py`
compares a regenerated folder with it and requires identical voxel data,
geometry and data types (file hashes may differ with gzip settings).

## ECPC-IDS fidelity

The original ECPC-IDS conversion code was not preserved, so `convert_ecpc.py`
was reconstructed from the data. Against the reference:

| Component | Exact | Remaining differences |
|---|---|---|
| Tumor labels | 132 / 134 cases | Case 067 (data type and header precision), case 140 |
| PET | 133 / 134 cases (values) | Case 140; headers of 40 cases differ in the fifth decimal of the slice spacing |
| CT | 67 / 134 cases | 48 cases: at most 1 HU in a few voxels (0.02% of all voxels); 18 cases: one edge slice is 0 in the reference; case 140 |

Case 140 has irregular DICOM slice spacing (4.2 mm) and was converted
differently in the original data. The differences are small but not zero, so
results obtained with regenerated ECPC-IDS data may differ slightly from the
paper.
