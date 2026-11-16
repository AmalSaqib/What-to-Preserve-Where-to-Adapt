#!/usr/bin/env python3
"""Convert the public UT-EndoMRI release into nnU-Net v1 task Task778_EndoMRI_T1FS.

Source: UT-EndoMRI.zip from https://doi.org/10.5281/zenodo.15750762
(MD5 7ace6e1b08efa10d0a1967073b0ba41c), extracted to ``source``. Files are
found by name anywhere below ``source``:

* images ``D{site}-{patient}_T1FS.nii.gz``;
* labels ``D{site}-{patient}_{structure}[_r{rater}].nii.gz`` with structure
  ``ut`` (uterus), ``ov`` (ovary), ``em`` (endometrioma) or ``cy`` (cyst).
  Site 1 provides up to three raters per structure (one file is misnamed
  ``D1-026_ut_re3``); site 2 provides one. Other files (``*_pat``) are ignored.

Each case ``D{site}-{patient}`` becomes ``EndoT1FS_{site}{patient}``:

* the T1FS image is copied unchanged to channel 0, and channel 1 is an
  all-zero volume with the same header;
* all label files of the case are resampled onto the image grid (nearest
  neighbour, only if their grid differs) and binarized. At every voxel the
  structure marked by the most files wins; ties are broken in the order
  endometrioma > cyst > ovary > uterus. Output labels are 1 uterus, 2 ovary,
  3 endometrioma, 4 cyst.

The case selection and the 83/20 training/held-out split are fixed by
``reference/Task778_EndoMRI_T1FS.csv`` (originally a seed-42 shuffle; see
``reference/Task778_EndoMRI_T1FS.fixed_split_80_20.json``).
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

HERE = Path(__file__).resolve().parent
LABELS = {"ut": 1, "ov": 2, "em": 3, "cy": 4}
PRIORITY = [3, 4, 2, 1]  # endometrioma > cyst > ovary > uterus
IMAGE = re.compile(r"D(\d+)-(\d+)_T1FS\.nii\.gz$")
LABEL = re.compile(r"D(\d+)-(\d+)_(ut|ov|em|cy)(?:_r\w*)?\.nii\.gz$")  # one rater file is named "_re3"


def reference_split(reference: Path) -> dict[str, str]:
    split = {}
    with reference.open() as handle:
        for row in csv.DictReader(handle):
            if row["split"].startswith("labels"):
                split[row["file"][: -len(".nii.gz")]] = row["split"][-2:]
    return split


def index_source(source: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    images, labels = {}, defaultdict(list)
    for path in sorted(source.rglob("*.nii.gz")):
        if match := IMAGE.match(path.name):
            images[f"EndoT1FS_{match[1]}{match[2]}"] = path
        elif match := LABEL.match(path.name):
            labels[f"EndoT1FS_{match[1]}{match[2]}"].append(path)
    return images, labels


def merge_labels(image: nib.Nifti1Image, label_files: list[Path]) -> np.ndarray:
    votes = np.zeros((len(LABELS) + 1,) + image.shape, dtype=np.int32)
    for path in label_files:
        label = nib.load(str(path))
        if label.shape != image.shape or not np.allclose(label.affine, image.affine):
            label = resample_from_to(label, (image.shape, image.affine), order=0)
        structure = LABEL.match(path.name)[3]
        votes[LABELS[structure]] += label.get_fdata() > 0

    best = votes[1:].max(axis=0)
    merged = np.zeros(image.shape, dtype=np.uint8)
    # Assign from lowest to highest priority so that higher priority wins ties.
    for value in reversed(PRIORITY):
        merged[(votes[value] == best) & (best > 0)] = value
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="extracted UT-EndoMRI folder")
    parser.add_argument("output", type=Path, help="output folder, e.g. .../nnUNet_raw_data/Task778_EndoMRI_T1FS")
    parser.add_argument("--reference", type=Path, default=HERE / "reference" / "Task778_EndoMRI_T1FS.csv")
    args = parser.parse_args()

    split = reference_split(args.reference)
    images, labels = index_source(args.source)
    missing = sorted(case for case in split if case not in images or not labels.get(case))
    if missing:
        raise SystemExit(f"Missing images or labels for {len(missing)} cases, e.g. {missing[:5]}")
    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (args.output / sub).mkdir(parents=True, exist_ok=True)

    for case, part in sorted(split.items()):
        image_path = images[case]
        image = nib.load(str(image_path))
        shutil.copyfile(image_path, args.output / f"images{part}" / f"{case}_0000.nii.gz")
        zeros = np.zeros_like(image.get_fdata())
        nib.save(nib.Nifti1Image(zeros, affine=image.affine, header=image.header),
                 str(args.output / f"images{part}" / f"{case}_0001.nii.gz"))
        merged = merge_labels(image, labels[case])
        nib.save(nib.Nifti1Image(merged, image.affine), str(args.output / f"labels{part}" / f"{case}.nii.gz"))

    shutil.copyfile(HERE / "reference" / "Task778_EndoMRI_T1FS.dataset.json", args.output / "dataset.json")
    counts = {part: sum(v == part for v in split.values()) for part in ("Tr", "Ts")}
    print(f"Wrote {counts['Tr']} training and {counts['Ts']} held-out cases to {args.output}")


if __name__ == "__main__":
    main()
