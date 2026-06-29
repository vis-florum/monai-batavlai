#!/usr/bin/env python
"""Convert LPD segmentation label .npy files to NRRD in parallel."""

import copy
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import nrrd
import numpy as np


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

SEGMENTATION_ROOTS = [
    # "/media/Store-SSD/Stembank/pine-LPDseg/5srcpos",
    # "/media/Store-SSD/Stembank/pine-LPDseg/9srcpos",
    "/media/Store-SSD/Stembank/pine-LPDseg/fullct/"
]

# Used only for geometry/header information. The log id is taken from the
# segmentation filename prefix, e.g. 000552_seg_joint_all_slices.npy -> 000552.
REFERENCE_NRRD_DIRS = [
    "/media/Store-SSD/Stembank/pine/LPDsample/test/labels/LPDsample-CT",
    # "/media/Store-SSD/Stembank/pine/LPDsample/test/labels/UNET_LPDsample-CT",
]

OVERWRITE = True
MAX_WORKERS = min(8, max(1, os.cpu_count() or 1))


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def discover_npy_files(roots):
    files = []
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.endswith(".npy"):
                    files.append(os.path.join(dirpath, filename))
    return sorted(files)


def log_id_from_filename(path):
    return os.path.basename(path).split("_", 1)[0]


def find_reference_nrrd(log_id, reference_dirs):
    for reference_dir in reference_dirs:
        reference_file = os.path.join(reference_dir, f"{log_id}.nrrd")
        if os.path.isfile(reference_file):
            return reference_file
    raise FileNotFoundError(f"No reference NRRD found for {log_id} in {reference_dirs}")


def load_binary_npy(path):
    label = np.load(path)
    label = np.swapaxes(label, 0, 1)
    label = (label > 0).astype(np.uint8)
    return np.ascontiguousarray(label)


def make_label_header(reference_header, label_shape):
    header = copy.deepcopy(reference_header)
    header["type"] = "uint8"
    header["dimension"] = 3
    header["sizes"] = np.asarray(label_shape, dtype=np.int64)
    return header


def convert_one(npy_file):
    nrrd_file = os.path.splitext(npy_file)[0] + ".nrrd"
    if os.path.exists(nrrd_file) and not OVERWRITE:
        return {"status": "skipped", "npy": npy_file, "nrrd": nrrd_file, "message": "exists"}

    log_id = log_id_from_filename(npy_file)
    reference_file = find_reference_nrrd(log_id, REFERENCE_NRRD_DIRS)
    _, reference_header = nrrd.read(reference_file)

    label = load_binary_npy(npy_file)
    header = make_label_header(reference_header, label.shape)
    nrrd.write(nrrd_file, label, header=header)

    return {"status": "converted", "npy": npy_file, "nrrd": nrrd_file, "message": ""}


def convert_one_safe(npy_file):
    try:
        return convert_one(npy_file)
    except Exception as exc:
        return {"status": "failed", "npy": npy_file, "nrrd": "", "message": repr(exc)}


def main():
    npy_files = discover_npy_files(SEGMENTATION_ROOTS)
    print(f"Found {len(npy_files)} .npy files")
    if not npy_files:
        return

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(convert_one_safe, npy_file) for npy_file in npy_files]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"{result['status']:9s} {result['npy']} {result['message']}")

    counts = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1

    print("\nSummary")
    for status in sorted(counts):
        print(f"{status}: {counts[status]}")

    failed = [result for result in results if result["status"] == "failed"]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
