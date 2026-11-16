#!/usr/bin/env python3
"""Convert the public ECPC-IDS release into nnU-Net v1 task Task555_ECPC.

Source: ECPC-IDS.zip from https://doi.org/10.6084/m9.figshare.23808258.v1
(MD5 9cac6378d62bef4d35fbc1ed2ed5cbd9), extracted so that ``source`` contains
``subset_A``. Only the 3D DICOM part (subset_A) is used.

For each case ``NNN`` (written as ``PETCT_NNN``):

* CT, PET and the tumor label are read as DICOM series and passed through
  NIfTI (single-precision geometry), as in the original conversion. The label
  series has the in-plane PET geometry (192 x 192, 3.65 mm) and is identical
  in ``part_CT`` and ``part_PET``; the ``part_CT`` copy is used.
* CT, PET and label series cover slightly different axial ranges. The output
  keeps the PET slices that lie inside both the CT and the label range.
* Channel 0 is CT resampled onto that grid with linear interpolation (value 0
  outside the CT field of view), stored as int32 Hounsfield units.
* Channel 1 is PET on the same slices, stored as float64 after DICOM
  rescaling.
* The label is resampled onto the grid (nearest neighbour) and stored as uint8
  (0 background, 1 tumor).

The case selection and the 108/26 training/held-out split are fixed by
``reference/Task555_ECPC.csv``. Of the 155 public cases, 134 are used.

Fidelity: the original conversion code was not preserved, so this script
was reconstructed from the data. Against the reference it reproduces the
label of 132/134 cases and the PET values of 133/134 cases exactly. CT is
identical in 67 cases, differs by at most 1 HU in a few voxels in 48, and in
18 cases one edge slice differs (filled with 0 in the reference). Case 140
has irregular DICOM slice spacing and is not reproduced; case 067 differs only
in header precision and label data type. PET headers of 40 cases differ in the
fifth decimal of the slice spacing. For exact data, contact the authors.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

HERE = Path(__file__).resolve().parent


def reference_split(reference: Path) -> dict[str, str]:
    split = {}
    with reference.open() as handle:
        for row in csv.DictReader(handle):
            if row["split"].startswith("labels"):
                split[row["file"][: -len(".nii.gz")]] = row["split"][-2:]
    return split


def read_series(folder: Path, nifti_geometry: bool = True) -> sitk.Image:
    """Read a DICOM series.

    With ``nifti_geometry`` the image is passed through a NIfTI file, which
    stores its geometry in single precision. The original conversion worked on
    NIfTI intermediates, and this rounding decides whether grid points lying
    exactly on the CT boundary count as inside it.
    """
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(folder)))
    image = reader.Execute()
    if nifti_geometry:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "series.nii.gz")
            sitk.WriteImage(image, path)
            image = sitk.ReadImage(path)
    return image


def axial_positions(image: sitk.Image) -> np.ndarray:
    return np.array([image.TransformIndexToPhysicalPoint((0, 0, k))[2] for k in range(image.GetSize()[2])])


def convert_case(source: Path, number: str) -> tuple[sitk.Image, sitk.Image, sitk.Image]:
    root = source / "subset_A"
    ct = read_series(root / "part_CT" / "ImageSet" / "DICOM" / number)
    pet = read_series(root / "part_PET" / "ImageSet" / "DICOM" / number)
    label = read_series(root / "part_CT" / "LabelSet" / "DICOM" / number)

    # Output slices: the PET slices inside both the CT and the label axial range.
    ct_z, pet_z, label_z = axial_positions(ct), axial_positions(pet), axial_positions(label)
    inside = lambda z, ref: (z >= ref.min() - 1e-2) & (z <= ref.max() + 1e-2)
    keep = np.flatnonzero(inside(pet_z, ct_z) & inside(pet_z, label_z))
    first, last = int(keep[0]), int(keep[-1]) + 1
    pet_crop = pet[:, :, first:last]

    # The grid takes its in-plane geometry and slice spacing from the label
    # series and its axial start from the first kept PET slice.
    grid = sitk.Image(pet_crop.GetSize(), sitk.sitkUInt8)
    grid.SetDirection(label.GetDirection())
    grid.SetSpacing(label.GetSpacing())
    grid.SetOrigin((label.GetOrigin()[0], label.GetOrigin()[1], pet_crop.GetOrigin()[2]))

    ct_out = sitk.Resample(ct, grid, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkInt32)

    pet_out = sitk.GetImageFromArray(sitk.GetArrayFromImage(pet_crop).astype(np.float64))
    pet_out.CopyInformation(pet_crop)

    label_out = sitk.Resample(label, grid, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    label_out = sitk.Cast(label_out > 0, sitk.sitkUInt8)
    return ct_out, pet_out, label_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="extracted ECPC-IDS folder containing subset_A")
    parser.add_argument("output", type=Path, help="output folder, e.g. .../nnUNet_raw_data/Task555_ECPC")
    parser.add_argument("--reference", type=Path, default=HERE / "reference" / "Task555_ECPC.csv")
    args = parser.parse_args()

    split = reference_split(args.reference)
    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (args.output / sub).mkdir(parents=True, exist_ok=True)

    for case, part in sorted(split.items()):
        ct, pet, label = convert_case(args.source, case.split("_")[1])
        sitk.WriteImage(ct, str(args.output / f"images{part}" / f"{case}_0000.nii.gz"))
        sitk.WriteImage(pet, str(args.output / f"images{part}" / f"{case}_0001.nii.gz"))
        sitk.WriteImage(label, str(args.output / f"labels{part}" / f"{case}.nii.gz"))

    shutil.copyfile(HERE / "reference" / "Task555_ECPC.dataset.json", args.output / "dataset.json")
    counts = {part: sum(v == part for v in split.values()) for part in ("Tr", "Ts")}
    print(f"Wrote {counts['Tr']} training and {counts['Ts']} held-out cases to {args.output}")


if __name__ == "__main__":
    main()
