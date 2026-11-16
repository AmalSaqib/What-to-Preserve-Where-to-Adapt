#!/usr/bin/env python3
"""Fingerprint converted nnU-Net v1 task folders.

For every image and label file this records the split, file name, SHA-256 of
the file, SHA-256 of the voxel array, array shape, voxel spacing, affine, data
type and (for labels) the label values present. File hashes depend on
compression settings; the voxel hash does not, so verification uses it. Conversion scripts are verified by regenerating a task
and comparing its manifest against the reference manifests in ``reference/``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

FIELDS = ["split", "file", "sha256", "data_sha256", "shape", "spacing", "affine", "dtype", "labels"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(path: Path, split: str) -> dict:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    labels = ""
    if split.startswith("labels"):
        labels = " ".join(str(int(v)) for v in np.unique(data))
    return {
        "split": split,
        "file": path.name,
        "sha256": sha256(path),
        "data_sha256": hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest(),
        "shape": "x".join(str(s) for s in data.shape),
        "spacing": " ".join(f"{z:.6g}" for z in image.header.get_zooms()),
        "affine": " ".join(f"{v:.6g}" for v in image.affine.ravel()),
        "dtype": str(data.dtype),
        "labels": labels,
    }


def manifest(task_dir: Path) -> list[dict]:
    rows = []
    for split in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        for path in sorted((task_dir / split).glob("*.nii.gz")):
            rows.append(describe(path, split))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_dir", type=Path, help="nnU-Net v1 task folder, e.g. .../Task666_UMD")
    parser.add_argument("output", type=Path, help="CSV file to write")
    args = parser.parse_args()
    rows = manifest(args.task_dir)
    if not rows:
        sys.exit(f"No NIfTI files found under {args.task_dir}")
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} files -> {args.output}")


if __name__ == "__main__":
    main()
