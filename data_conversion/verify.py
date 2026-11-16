#!/usr/bin/env python3
"""Compare a regenerated nnU-Net task folder against a reference manifest.

Every file in the reference must exist in the same split with identical voxel
data (``data_sha256``), shape, data type, spacing and affine. File-level
SHA-256 values are reported but not required to match, because they depend on
gzip settings. Exit status is 0 only if every file matches.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from make_manifest import manifest

CHECKED = ("data_sha256", "shape", "dtype", "spacing", "affine")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_dir", type=Path, help="regenerated nnU-Net v1 task folder")
    parser.add_argument("reference", type=Path, help="reference manifest CSV")
    args = parser.parse_args()

    with args.reference.open() as handle:
        expected = {(row["split"], row["file"]): row for row in csv.DictReader(handle)}
    found = {(row["split"], row["file"]): row for row in manifest(args.task_dir)}

    missing = sorted(set(expected) - set(found))
    extra = sorted(set(found) - set(expected))
    mismatched = {}
    identical_files = 0
    for key in sorted(set(expected) & set(found)):
        fields = [f for f in CHECKED if expected[key][f] != found[key][f]]
        if fields:
            mismatched[key] = fields
        identical_files += expected[key]["sha256"] == found[key]["sha256"]

    print(f"reference files: {len(expected)}  regenerated: {len(found)}")
    print(f"missing: {len(missing)}  unexpected: {len(extra)}  mismatched: {len(mismatched)}")
    print(f"byte-identical files: {identical_files}")
    for label, items in (("missing", missing), ("unexpected", extra)):
        for split, name in items[:10]:
            print(f"  {label}: {split}/{name}")
    for (split, name), fields in list(mismatched.items())[:10]:
        print(f"  mismatch: {split}/{name}: {', '.join(fields)}")
    ok = not (missing or extra or mismatched)
    print("MATCH" if ok else "DIFFERENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
