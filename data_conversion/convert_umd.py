#!/usr/bin/env python3
"""Convert the public UMD release into nnU-Net v1 task Task666_UMD.

Source: UMD.zip from https://doi.org/10.6084/m9.figshare.23541312.v3
(MD5 f52985246c363a7fb9c8f6ae33dcde84), extracted so that ``source`` contains
the folders ``UMD_221129_001`` ... ``UMD_221129_300``.

Each case ``UMD_221129_NNN`` becomes ``UMD_NNN``:

* ``UMD_221129_NNN_t2.nii.gz`` is copied unchanged to channel 0;
* channel 1 is an all-zero volume with the same header and data type, so the
  input matches the two-channel (CT, PET) layout of ECPC-IDS;
* ``UMD_221129_NNN_seg.nii.gz`` is copied unchanged as the label
  (1 uterine wall, 2 uterine cavity, 3 myoma, 4 nabothian cyst).

34 cases of the public release are not used because their segmentation file
is not a readable NIfTI (33 are named ``*_seq.nii.gz`` and one
``UMD_2221129_037_seg.nii.gz``; none is gzip-compressed). The remaining 266
cases are split into 214 training and 52 held-out cases; the split was drawn
at random once and is fixed by ``reference/Task666_UMD.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent


def reference_split(reference: Path) -> dict[str, str]:
    """Map case ID (e.g. UMD_001) to 'Tr' or 'Ts' from a reference manifest."""
    split = {}
    with reference.open() as handle:
        for row in csv.DictReader(handle):
            if row["split"].startswith("labels"):
                split[row["file"][: -len(".nii.gz")]] = row["split"][-2:]
    return split


def write_zero_channel(image_path: Path, output: Path) -> None:
    image = nib.load(str(image_path))
    zeros = np.zeros(image.shape, dtype=image.get_data_dtype())
    nib.save(nib.Nifti1Image(zeros, image.affine, image.header), str(output))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="extracted UMD folder")
    parser.add_argument("output", type=Path, help="output folder, e.g. .../nnUNet_raw_data/Task666_UMD")
    parser.add_argument("--reference", type=Path, default=HERE / "reference" / "Task666_UMD.csv")
    args = parser.parse_args()

    split = reference_split(args.reference)
    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (args.output / sub).mkdir(parents=True, exist_ok=True)

    for case, part in sorted(split.items()):
        number = case.split("_")[1]
        folder = args.source / f"UMD_221129_{number}"
        image = folder / f"UMD_221129_{number}_t2.nii.gz"
        label = folder / f"UMD_221129_{number}_seg.nii.gz"
        shutil.copyfile(image, args.output / f"images{part}" / f"{case}_0000.nii.gz")
        write_zero_channel(image, args.output / f"images{part}" / f"{case}_0001.nii.gz")
        shutil.copyfile(label, args.output / f"labels{part}" / f"{case}.nii.gz")

    shutil.copyfile(HERE / "reference" / "Task666_UMD.dataset.json", args.output / "dataset.json")
    counts = {part: sum(v == part for v in split.values()) for part in ("Tr", "Ts")}
    print(f"Wrote {counts['Tr']} training and {counts['Ts']} held-out cases to {args.output}")


if __name__ == "__main__":
    main()
