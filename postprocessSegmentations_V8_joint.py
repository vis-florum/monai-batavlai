#!/usr/bin/env python
"""
Final region-level dice analysis for CT, CT-new, 5 source-position LPD
segmentations, and 10 source-position LPD segmentations.

This script evaluates every single GT knot group in the test logs, stores
raw and normalized dice profiles, ranks the best/worst groups, and writes
old-style contour overlays for the selected groups.
"""

import csv
import glob
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# Keep Matplotlib from trying to write into ~/.config in restricted shells.
os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib-cache"))

import matplotlib.pyplot as plt
import nrrd
import numpy as np


# ---------------------------------------------------------------------------
# Preamble: edit these settings for new runs or additional plots
# ---------------------------------------------------------------------------

ROOT_CT = "/media/Store-SSD/Stembank/pine/LPDsample/"
ROOT_SEG_5 = "/media/Store-SSD/Stembank/pine-LPDseg/5srcpos/"
ROOT_SEG_10 = "/media/Store-SSD/Stembank/pine-LPDseg/10srcpos/"
ROOT_SEG_FULLCT = "/media/Store-SSD/Stembank/pine-LPDseg/fullct/"
ROOT_OUT = "/home/aime/monai/Postprocessing/LPDseg_final"

DATASETS = ["test"]
PROFILE_LENGTH = 101
NO_ERROR_BARS = 6

# Set this to False if source labels changed and profiles must be recomputed.
REUSE_PROFILE_CACHE = True

# Previous output folders can be checked before recomputing profiles if their
# variant names match this script.
EXTERNAL_PROFILE_CACHE_ROOTS = [
]

MAX_WORKERS = min(6, max(1, os.cpu_count() or 1))
TOP_N = 5
RANK_BY_VARIANTS = ["5src-LPD-jnt", "10src-LPD-jnt"]

# Profiles included in figures. Calculations and CSV files still include all
# segmentation variants.
PROFILE_PLOT_VARIANTS = [
    # "CT",
    "CT-new",
    "5src-LPD-seq",
    "5src-LPD-jnt",
    "10src-LPD-seq",
    "10src-LPD-jnt",
]

GENERATE_SIZE_HISTOGRAM = True
HIST_NO_BINS = 15

KNOT_REGIONS = {
    "Start": (0.0, 0.2),
    "Mid": (0.2, 0.8),
    "End": (0.8, 1.0),
    "Total": (0.0, 1.0),
}

# Increment the version if the volume-Dice calculation is changed.
REUSE_VOLUME_METRIC_CACHE = True
VOLUME_METRIC_CACHE_VERSION = 1

GENERATE_PROFILE_PLOTS = False
GENERATE_CONTOURS = True
CUSTOM_CONTOURS_ONLY = True
SLICE_PAD = 3
# CROP = (35, 30, 30, 50)  # xs, xe, ys, ye, using the same style as V4/V5
KNOT_CROP_PAD = 5
LINEWIDTH = 1.0

# Add manually requested knot groups here. These are plotted in addition to
# the automatically selected best/worst groups.
EXTRA_CONTOUR_GROUPS = [
    {"dataset": "test", "log_id": "002753", "region_nr": 5},
]

SEG_VARIANTS = ["CT-new", "5src-LPD-seq", "5src-LPD-jnt", "10src-LPD-seq", "10src-LPD-jnt"]
ORDERED_VARIANTS = ["GT", "CT"] + SEG_VARIANTS

SEG_VARIANT_DIRS = {
    "CT-new": {"root": ROOT_SEG_FULLCT, "folder": ""},
    "5src-LPD-seq": {"root": ROOT_SEG_5, "folder": "sequential"},
    "5src-LPD-jnt": {"root": ROOT_SEG_5, "folder": "joint"},
    "10src-LPD-seq": {"root": ROOT_SEG_10, "folder": "sequential"},
    "10src-LPD-jnt": {"root": ROOT_SEG_10, "folder": "joint"},
}

VARIANT_LABELS = {
    "GT": "GT",
    # "CT": "CT",
    "CT-new": "CT",
    "5src-LPD-seq": "5 src. sequential",
    "5src-LPD-jnt": "5 src. joint",
    "10src-LPD-seq": "10 src. sequential",
    "10src-LPD-jnt": "10 src. joint",
}

# 5src uses distinct colors from 10src in the all-source-position comparison.
PLOT_STYLE = {
    "CT-new": {"color": "blue", "linestyle": "dashed"},
    # "CT-new": {"color": "purple", "linestyle": "dashed"},
    "5src-LPD-seq": {"color": "orange", "linestyle": "solid"},
    "5src-LPD-jnt": {"color": "cyan", "linestyle": "solid"},
    "10src-LPD-seq": {"color": "green", "linestyle": "solid"},
    "10src-LPD-jnt": {"color": "red", "linestyle": "solid"},
}

CONTOUR_COLORS = {
    "GT": "black",
    "CT-new": (0.1, 0.1, 1.0, 1.0),
    # "CT-new": "purple",
    "5src-LPD-seq": "orange",
    "5src-LPD-jnt": "cyan",
    "10src-LPD-seq": (0.1, 1.0, 0.1, 1.0),
    "10src-LPD-jnt": "red",
}

# Edit these groups when you want different contour comparisons.
CONTOUR_PLOT_GROUPS = {
    # "ct_ctnew_10src": ["GT", "CT", "CT-new", "10src-LPD-seq", "10src-LPD-jnt"],
    "ct_ctnew_10src": ["GT", "CT-new", "10src-LPD-seq", "10src-LPD-jnt"],
    "10src_joint_vs_5src": ["GT", "10src-LPD-jnt", "5src-LPD-seq", "5src-LPD-jnt"],
}


# ---------------------------------------------------------------------------
# File and volume helpers
# ---------------------------------------------------------------------------

def find_label_file(label_dir, log_id):
    patterns = [
        os.path.join(label_dir, f"{log_id}*.nrrd"),
        os.path.join(label_dir, f"{log_id}*.npy"),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No label file found for {log_id} in {label_dir}")
    return files[0]


def find_segmentation_label_file(seg_variant_dir, log_id):
    patterns = [
        os.path.join(seg_variant_dir, f"{log_id}*.nrrd"),
        os.path.join(seg_variant_dir, f"{log_id}*.npy"),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No segmentation file found for {log_id} in {seg_variant_dir}")
    return files[0]


def list_log_ids(label_dir):
    names = []
    for fname in os.listdir(label_dir):
        if fname.endswith(".nrrd") or fname.endswith(".npy"):
            names.append(os.path.splitext(fname)[0])
    return sorted(set(names))


def resolve_segmentation_dir(seg_variant_dirs, variant, dataset):
    variant_config = seg_variant_dirs[variant]
    root_seg = variant_config["root"]
    variant_folder = variant_config["folder"]
    candidate_dirs = [
        os.path.join(root_seg, variant_folder, dataset, "labels"),
        os.path.join(root_seg, variant_folder, dataset),
        os.path.join(root_seg, variant_folder),
    ]
    variant_dir = next((path for path in candidate_dirs if os.path.isdir(path)), None)
    if variant_dir is None:
        raise FileNotFoundError(f"No segmentation directory found for '{variant}' in {candidate_dirs}")
    return variant_dir


def load_volume(volume_file):
    if volume_file.endswith(".nrrd"):
        volume, _ = nrrd.read(volume_file)
        volume = np.asarray(volume)
    elif volume_file.endswith(".npy"):
        volume = np.load(volume_file)
        volume = np.swapaxes(volume, 0, 1)
    else:
        raise ValueError(f"Unsupported volume format: {volume_file}")
    return volume


def load_label_binary(label_file):
    label = load_volume(label_file)
    label = np.asarray(label)
    return (label > 0).astype(np.uint8)


# def crop_slice(volume, xs, xe, ys, ye, idx):
#     x_stop = -xe if xe else None
#     y_stop = -ye if ye else None
#     return volume[xs:x_stop, ys:y_stop, idx]

def crop_slice(volume, x_start, x_stop, y_start, y_stop, idx):
    return volume[
        x_start:x_stop,
        y_start:y_stop,
        idx,
    ]

def knot_group_crop(
    image,
    slice_start,
    slice_end,
    padding=0,
):
    knot_volume = image[
        :,
        :,
        slice_start:slice_end + 1,
    ]

    foreground = knot_volume > 70
    foreground_xy = np.any(foreground, axis=2)

    coordinates = np.argwhere(foreground_xy)

    if coordinates.size == 0:
        return (
            0,
            image.shape[0],
            0,
            image.shape[1],
        )

    x_start, y_start = coordinates.min(axis=0)
    x_stop, y_stop = coordinates.max(axis=0) + 1

    return (
        max(int(x_start) - padding, 0),
        min(int(x_stop) + padding, image.shape[0]),
        max(int(y_start) - padding, 0),
        min(int(y_stop) + padding, image.shape[1]),
    )


# ---------------------------------------------------------------------------
# Dice profiles and region extraction
# ---------------------------------------------------------------------------

def dice_profile(pred_label, gt_label):
    pred = pred_label.astype(bool)
    gt = gt_label.astype(bool)
    scores = np.empty(gt.shape[2], dtype=np.float32)
    for idx in range(gt.shape[2]):
        pred_slice = pred[:, :, idx]
        gt_slice = gt[:, :, idx]
        denom = pred_slice.sum() + gt_slice.sum()
        if denom == 0:
            scores[idx] = np.nan
        else:
            scores[idx] = 2.0 * np.logical_and(pred_slice, gt_slice).sum() / denom
    return scores


def connected_regions_1d(mask):
    regions = []
    in_region = False
    start = None
    region_nr = 0
    for idx, value in enumerate(mask):
        if value and not in_region:
            in_region = True
            start = idx
        if in_region and (not value or idx == len(mask) - 1):
            end = idx if value and idx == len(mask) - 1 else idx - 1
            region_nr += 1
            regions.append({"region_nr": region_nr, "slice_start": int(start), "slice_end": int(end)})
            in_region = False
    return regions


def interp_profile(profile, newlength=101):
    if len(profile) == 0:
        return np.full(newlength, np.nan, dtype=np.float32)
    if len(profile) == 1:
        return np.full(newlength, profile[0], dtype=np.float32)
    return np.interp(
        np.linspace(0, 1, newlength),
        np.linspace(0, 1, len(profile)),
        profile,
    ).astype(np.float32)


def safe_float(value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, np.floating) and np.isnan(value):
        return ""
    return float(value)


def profile_stats(profile):
    if len(profile) == 0 or np.all(np.isnan(profile)):
        return {"mean": np.nan, "median": np.nan, "min": np.nan, "std": np.nan}
    return {
        "mean": float(np.nanmean(profile)),
        "median": float(np.nanmedian(profile)),
        "min": float(np.nanmin(profile)),
        "std": float(np.nanstd(profile)),
    }


# ---------------------------------------------------------------------------
# Cache handling
# ---------------------------------------------------------------------------

def internal_profile_cache_file(root_out, dataset, log_id):
    return os.path.join(root_out, "cache", "profiles", dataset, f"{log_id}_dice_profiles.npy")


def external_profile_cache_file(cache_root, dataset, log_id):
    return os.path.join(cache_root, dataset, "dice", f"{log_id}_dice_profiles.npy")


def load_cached_profiles(root_out, external_roots, dataset, log_id, required_variants):
    if not REUSE_PROFILE_CACHE:
        return {}

    profiles = {}
    cache_files = [internal_profile_cache_file(root_out, dataset, log_id)]
    cache_files.extend(external_profile_cache_file(root, dataset, log_id) for root in external_roots)

    for cache_file in cache_files:
        if not os.path.isfile(cache_file):
            continue
        cached = np.load(cache_file, allow_pickle=True)[()]
        for variant in required_variants:
            if variant in cached and variant not in profiles:
                profiles[variant] = np.asarray(cached[variant], dtype=np.float32)
        if all(variant in profiles for variant in required_variants):
            break

    return profiles


def save_cached_profiles(root_out, dataset, log_id, profiles):
    cache_file = internal_profile_cache_file(root_out, dataset, log_id)
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    np.save(cache_file, {variant: np.asarray(profile) for variant, profile in profiles.items()})

def internal_volume_metric_cache_file(root_out, dataset, log_id):
    return os.path.join(
        root_out,
        "cache",
        "volume_metrics",
        dataset,
        f"{log_id}_volume_metrics.npy",
    )


def region_signature(regions):
    return [
        (
            int(region["region_nr"]),
            int(region["slice_start"]),
            int(region["slice_end"]),
        )
        for region in regions
    ]


def load_cached_volume_metrics(
    root_out,
    dataset,
    log_id,
    regions,
    required_variants,
):
    if not REUSE_VOLUME_METRIC_CACHE:
        return {}

    cache_file = internal_volume_metric_cache_file(
        root_out,
        dataset,
        log_id,
    )
    if not os.path.isfile(cache_file):
        return {}

    try:
        cached = np.load(cache_file, allow_pickle=True)[()]
    except (OSError, ValueError, EOFError):
        return {}

    if cached.get("cache_version") != VOLUME_METRIC_CACHE_VERSION:
        return {}

    cached_signature = [
        tuple(item)
        for item in cached.get("region_signature", [])
    ]
    if cached_signature != region_signature(regions):
        return {}

    required_region_nrs = {
        int(region["region_nr"])
        for region in regions
    }

    cached_metrics = cached.get("metrics", {})
    valid_metrics = {}

    for variant in required_variants:
        variant_metrics = cached_metrics.get(variant)
        if not isinstance(variant_metrics, dict):
            continue

        full_volume = variant_metrics.get("full_volume")
        region_metrics = variant_metrics.get("regions", {})

        if not isinstance(full_volume, dict):
            continue
        if not isinstance(region_metrics, dict):
            continue

        region_metrics = {
            int(key): value
            for key, value in region_metrics.items()
        }

        if set(region_metrics) != required_region_nrs:
            continue

        valid_metrics[variant] = {
            "full_volume": full_volume,
            "regions": region_metrics,
        }

    return valid_metrics


def save_cached_volume_metrics(
    root_out,
    dataset,
    log_id,
    regions,
    metrics,
):
    cache_file = internal_volume_metric_cache_file(
        root_out,
        dataset,
        log_id,
    )
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    np.save(
        cache_file,
        {
            "cache_version": VOLUME_METRIC_CACHE_VERSION,
            "region_signature": region_signature(regions),
            "metrics": metrics,
        },
    )


def binary_volume_metrics(pred_label, gt_label):
    pred = np.asarray(pred_label, dtype=bool)
    gt = np.asarray(gt_label, dtype=bool)

    pred_voxels = int(pred.sum())
    gt_voxels = int(gt.sum())
    intersection_voxels = int(
        np.logical_and(pred, gt).sum()
    )

    denominator = pred_voxels + gt_voxels
    volume_dice = (
        np.nan
        if denominator == 0
        else 2.0 * intersection_voxels / denominator
    )

    return {
        "intersection_voxels": intersection_voxels,
        "pred_voxels": pred_voxels,
        "gt_voxels": gt_voxels,
        "volume_dice": (
            float(volume_dice)
            if not np.isnan(volume_dice)
            else np.nan
        ),
    }


def compute_variant_volume_metrics(
    pred_label,
    gt_label,
    regions,
):
    metrics = {
        "full_volume": binary_volume_metrics(
            pred_label,
            gt_label,
        ),
        "regions": {},
    }

    for region in regions:
        region_nr = int(region["region_nr"])
        region_slice = slice(
            int(region["slice_start"]),
            int(region["slice_end"]) + 1,
        )

        metrics["regions"][region_nr] = binary_volume_metrics(
            pred_label[:, :, region_slice],
            gt_label[:, :, region_slice],
        )

    return metrics

# ---------------------------------------------------------------------------
# Per-log worker
# ---------------------------------------------------------------------------

def process_log(task):
    dataset = task["dataset"]
    log_id = task["log_id"]
    gt_dir = task["gt_dir"]
    ct_dir = task["ct_dir"]
    root_out = task["root_out"]
    external_cache_roots = task["external_cache_roots"]
    seg_variant_dirs = task["seg_variant_dirs"]
    ordered_variants = task["ordered_variants"]
    profile_length = task["profile_length"]

    evaluated_variants = [
        variant
        for variant in ordered_variants
        if variant != "GT"
    ]

    profiles = load_cached_profiles(
        root_out,
        external_cache_roots,
        dataset,
        log_id,
        ordered_variants,
    )

    gt_label = load_label_binary(
        find_label_file(gt_dir, log_id)
    )

    if "GT" not in profiles:
        profiles["GT"] = dice_profile(
            gt_label,
            gt_label,
        )

    gt_indicator = (
        np.nan_to_num(profiles["GT"], nan=0.0) > 0
    )
    regions = connected_regions_1d(gt_indicator)

    volume_metrics = load_cached_volume_metrics(
        root_out,
        dataset,
        log_id,
        regions,
        evaluated_variants,
    )

    def load_prediction(variant):
        if variant == "CT":
            label_file = find_label_file(
                ct_dir,
                log_id,
            )
        else:
            variant_dir = resolve_segmentation_dir(
                seg_variant_dirs,
                variant,
                dataset,
            )
            label_file = find_segmentation_label_file(
                variant_dir,
                log_id,
            )

        pred_label = load_label_binary(label_file)

        if pred_label.shape != gt_label.shape:
            raise ValueError(
                f"Shape mismatch for "
                f"{dataset}/{log_id}/{variant}: "
                f"prediction {pred_label.shape}, "
                f"GT {gt_label.shape}"
            )

        return pred_label

    for variant in evaluated_variants:
        needs_profile = variant not in profiles
        needs_volume_metrics = variant not in volume_metrics

        if not needs_profile and not needs_volume_metrics:
            continue

        pred_label = load_prediction(variant)

        if needs_profile:
            profiles[variant] = dice_profile(
                pred_label,
                gt_label,
            )

        if needs_volume_metrics:
            volume_metrics[variant] = (
                compute_variant_volume_metrics(
                    pred_label,
                    gt_label,
                    regions,
                )
            )

    save_cached_profiles(
        root_out,
        dataset,
        log_id,
        profiles,
    )
    save_cached_volume_metrics(
        root_out,
        dataset,
        log_id,
        regions,
        volume_metrics,
    )

    table_rows = []
    raw_rows = []
    normed_rows = []
    # log_volume_rows = []

    for region in regions:
        region_nr = int(region["region_nr"])
        slice_start = int(region["slice_start"])
        slice_end = int(region["slice_end"])

        region_slice = slice(
            slice_start,
            slice_end + 1,
        )
        length_slices = slice_end - slice_start + 1
        gt_region = gt_label[:, :, region_slice]

        table_row = {
            "dataset": dataset,
            "log_id": log_id,
            "region_nr": region_nr,
            "slice_start": slice_start,
            "slice_end": slice_end,
            "length_slices": length_slices,
            "gt_voxels": int(gt_region.sum()),
        }

        for variant in evaluated_variants:
            raw_profile = np.asarray(
                profiles[variant][region_slice],
                dtype=np.float32,
            )
            normed_profile = interp_profile(
                raw_profile,
                profile_length,
            )

            stats = profile_stats(raw_profile)
            for stat_name, value in stats.items():
                table_row[
                    f"{variant}_{stat_name}_dice"
                ] = safe_float(value)

            region_volume = (
                volume_metrics[variant]["regions"][region_nr]
            )

            table_row[
                f"{variant}_volume_intersection_voxels"
            ] = int(region_volume["intersection_voxels"])
            table_row[
                f"{variant}_volume_pred_voxels"
            ] = int(region_volume["pred_voxels"])
            table_row[
                f"{variant}_volume_gt_voxels"
            ] = int(region_volume["gt_voxels"])
            table_row[
                f"{variant}_volume_dice"
            ] = safe_float(region_volume["volume_dice"])

            for offset, value in enumerate(raw_profile):
                raw_rows.append({
                    "dataset": dataset,
                    "log_id": log_id,
                    "region_nr": region_nr,
                    "slice_idx": slice_start + offset,
                    "relative_idx": safe_float(
                        offset / max(length_slices - 1, 1)
                    ),
                    "variant": variant,
                    "dice": safe_float(value),
                })

            for point_idx, value in enumerate(
                normed_profile
            ):
                normed_rows.append({
                    "dataset": dataset,
                    "log_id": log_id,
                    "region_nr": region_nr,
                    "point_idx": point_idx,
                    "relative_idx": safe_float(
                        point_idx
                        / max(profile_length - 1, 1)
                    ),
                    "variant": variant,
                    "dice": safe_float(value),
                })

        table_rows.append(table_row)

    log_volume_rows = []

    for variant in evaluated_variants:
        full_volume = (
            volume_metrics[variant]["full_volume"]
        )

        log_volume_rows.append({
            "dataset": dataset,
            "log_id": log_id,
            "variant": variant,
            "volume_intersection_voxels": int(
                full_volume["intersection_voxels"]
            ),
            "volume_pred_voxels": int(
                full_volume["pred_voxels"]
            ),
            "volume_gt_voxels": int(
                full_volume["gt_voxels"]
            ),
            "volume_dice": safe_float(
                full_volume["volume_dice"]
            ),
        })

    return {
        "dataset": dataset,
        "log_id": log_id,
        "table_rows": table_rows,
        "raw_rows": raw_rows,
        "normed_rows": normed_rows,
        "log_volume_rows": log_volume_rows,
    }

def assign_knot_size_categories(
    table_rows,
    raw_rows,
    normed_rows,
):
    threshold_rows = []
    size_lookup = {}

    datasets = sorted({
        row["dataset"]
        for row in table_rows
    })

    for dataset in datasets:
        dataset_rows = [
            row
            for row in table_rows
            if row["dataset"] == dataset
        ]

        volumes = np.asarray(
            [
                int(row["gt_voxels"])
                for row in dataset_rows
            ],
            dtype=np.float64,
        )

        if len(volumes) == 0:
            continue

        lower_threshold, upper_threshold = (
            np.percentile(
                volumes,
                [100.0 / 3.0, 200.0 / 3.0],
            )
        )

        counts = {
            "small": 0,
            "medium": 0,
            "large": 0,
        }

        for row in dataset_rows:
            volume = int(row["gt_voxels"])

            if volume <= lower_threshold:
                size_category = "small"
            elif volume <= upper_threshold:
                size_category = "medium"
            else:
                size_category = "large"

            row["size_category"] = size_category
            counts[size_category] += 1

            size_lookup[
                (
                    row["dataset"],
                    row["log_id"],
                    int(row["region_nr"]),
                )
            ] = size_category

        threshold_rows.append({
            "dataset": dataset,
            "n_knot_groups": int(len(volumes)),
            "lower_tercile_threshold_voxels": float(
                lower_threshold
            ),
            "upper_tercile_threshold_voxels": float(
                upper_threshold
            ),
            "n_small": counts["small"],
            "n_medium": counts["medium"],
            "n_large": counts["large"],
        })

    for profile_rows in (raw_rows, normed_rows):
        for row in profile_rows:
            key = (
                row["dataset"],
                row["log_id"],
                int(row["region_nr"]),
            )
            row["size_category"] = size_lookup[key]

    return threshold_rows


def knot_region_mask(
    relative_positions,
    start,
    end,
):
    relative_positions = np.asarray(
        relative_positions,
        dtype=np.float64,
    )

    if end >= 1.0:
        return (
            (relative_positions >= start)
            & (relative_positions <= end)
        )

    return (
        (relative_positions >= start)
        & (relative_positions < end)
    )


def calculate_knot_region_scores(
    profile_rows,
    profile_type,
    knot_regions,
):
    grouped = {}

    for row in profile_rows:
        if row["dice"] == "":
            continue

        key = (
            row["dataset"],
            row["log_id"],
            int(row["region_nr"]),
            row["size_category"],
            row["variant"],
        )

        grouped.setdefault(key, []).append(
            (
                float(row["relative_idx"]),
                float(row["dice"]),
            )
        )

    score_rows = []

    for key, values in grouped.items():
        (
            dataset,
            log_id,
            region_nr,
            size_category,
            variant,
        ) = key

        relative_positions = np.asarray(
            [value[0] for value in values],
            dtype=np.float64,
        )
        dice_values = np.asarray(
            [value[1] for value in values],
            dtype=np.float64,
        )

        for knot_region, bounds in knot_regions.items():
            start, end = bounds

            mask = knot_region_mask(
                relative_positions,
                start,
                end,
            )
            selected = dice_values[mask]

            mean_dice = (
                np.nan
                if len(selected) == 0
                else float(np.nanmean(selected))
            )

            score_rows.append({
                "dataset": dataset,
                "log_id": log_id,
                "region_nr": region_nr,
                "size_category": size_category,
                "variant": variant,
                "profile_type": profile_type,
                "knot_region": knot_region,
                "relative_start": float(start),
                "relative_end": float(end),
                "mean_dice": safe_float(mean_dice),
                "n_profile_points": int(
                    np.sum(~np.isnan(selected))
                ),
            })

    return sorted(
        score_rows,
        key=lambda row: (
            row["profile_type"],
            row["dataset"],
            row["log_id"],
            int(row["region_nr"]),
            row["variant"],
            row["knot_region"],
        ),
    )


def aggregate_knot_region_scores(
    region_score_rows,
):
    grouped = {}

    for row in region_score_rows:
        if row["mean_dice"] == "":
            continue

        size_groups = [
            "all",
            row["size_category"],
        ]
        scopes = [
            (
                "log",
                row["dataset"],
                row["log_id"],
            ),
            (
                "all_logs",
                row["dataset"],
                "ALL",
            ),
        ]

        for aggregate_scope, dataset, log_id in scopes:
            for size_category in size_groups:
                key = (
                    aggregate_scope,
                    dataset,
                    log_id,
                    size_category,
                    row["profile_type"],
                    row["variant"],
                    row["knot_region"],
                )

                grouped.setdefault(key, []).append(
                    float(row["mean_dice"])
                )

    summary_rows = []

    for key, values in grouped.items():
        (
            aggregate_scope,
            dataset,
            log_id,
            size_category,
            profile_type,
            variant,
            knot_region,
        ) = key

        values_array = np.asarray(
            values,
            dtype=np.float64,
        )

        summary_rows.append({
            "aggregate_scope": aggregate_scope,
            "dataset": dataset,
            "log_id": log_id,
            "size_category": size_category,
            "profile_type": profile_type,
            "variant": variant,
            "knot_region": knot_region,
            "mean_dice": safe_float(
                float(np.nanmean(values_array))
            ),
            "median_dice": safe_float(
                float(np.nanmedian(values_array))
            ),
            "std_dice": safe_float(
                float(np.nanstd(values_array))
            ),
            "n_knot_groups": int(
                np.sum(~np.isnan(values_array))
            ),
        })

    return sorted(
        summary_rows,
        key=lambda row: (
            row["aggregate_scope"],
            row["dataset"],
            row["log_id"],
            row["size_category"],
            row["profile_type"],
            row["variant"],
            row["knot_region"],
        ),
    )


def add_region_scores_to_knot_table(
    table_rows,
    region_score_rows,
):
    table_lookup = {
        (
            row["dataset"],
            row["log_id"],
            int(row["region_nr"]),
        ): row
        for row in table_rows
    }

    for score_row in region_score_rows:
        key = (
            score_row["dataset"],
            score_row["log_id"],
            int(score_row["region_nr"]),
        )

        column_name = (
            f'{score_row["variant"]}_'
            f'{score_row["profile_type"]}_'
            f'{score_row["knot_region"].lower()}_'
            f'mean_dice'
        )

        table_lookup[key][column_name] = (
            score_row["mean_dice"]
        )


def aggregate_volume_dice(
    table_rows,
    log_volume_rows,
    evaluated_variants,
):
    grouped = {}

    def add_counts(
        key,
        intersection_voxels,
        pred_voxels,
        gt_voxels,
        log_id,
        knot_group_increment,
    ):
        accumulator = grouped.setdefault(
            key,
            {
                "intersection_voxels": 0,
                "pred_voxels": 0,
                "gt_voxels": 0,
                "log_ids": set(),
                "n_knot_groups": 0,
            },
        )

        accumulator["intersection_voxels"] += int(
            intersection_voxels
        )
        accumulator["pred_voxels"] += int(
            pred_voxels
        )
        accumulator["gt_voxels"] += int(
            gt_voxels
        )
        accumulator["log_ids"].add(log_id)
        accumulator["n_knot_groups"] += int(
            knot_group_increment
        )

    # Exact full-volume Dice for each log and all logs.
    for row in log_volume_rows:
        for aggregate_scope, output_log_id in [
            ("log", row["log_id"]),
            ("all_logs", "ALL"),
        ]:
            key = (
                "full_volume",
                aggregate_scope,
                row["dataset"],
                output_log_id,
                "all",
                row["variant"],
            )

            add_counts(
                key,
                row["volume_intersection_voxels"],
                row["volume_pred_voxels"],
                row["volume_gt_voxels"],
                row["log_id"],
                0,
            )

    # Pooled Dice within GT knot-group slice spans.
    for row in table_rows:
        for variant in evaluated_variants:
            for aggregate_scope, output_log_id in [
                ("log", row["log_id"]),
                ("all_logs", "ALL"),
            ]:
                for size_category in [
                    "all",
                    row["size_category"],
                ]:
                    key = (
                        "gt_knot_group_spans",
                        aggregate_scope,
                        row["dataset"],
                        output_log_id,
                        size_category,
                        variant,
                    )

                    add_counts(
                        key,
                        row[
                            f"{variant}_"
                            "volume_intersection_voxels"
                        ],
                        row[
                            f"{variant}_"
                            "volume_pred_voxels"
                        ],
                        row[
                            f"{variant}_"
                            "volume_gt_voxels"
                        ],
                        row["log_id"],
                        1,
                    )

    volume_rows = []

    for key, counts in grouped.items():
        (
            evaluation_domain,
            aggregate_scope,
            dataset,
            log_id,
            size_category,
            variant,
        ) = key

        denominator = (
            counts["pred_voxels"]
            + counts["gt_voxels"]
        )

        volume_dice = (
            np.nan
            if denominator == 0
            else (
                2.0
                * counts["intersection_voxels"]
                / denominator
            )
        )

        volume_rows.append({
            "evaluation_domain": evaluation_domain,
            "aggregate_scope": aggregate_scope,
            "dataset": dataset,
            "log_id": log_id,
            "size_category": size_category,
            "variant": variant,
            "volume_intersection_voxels": (
                counts["intersection_voxels"]
            ),
            "volume_pred_voxels": (
                counts["pred_voxels"]
            ),
            "volume_gt_voxels": (
                counts["gt_voxels"]
            ),
            "volume_dice": safe_float(volume_dice),
            "n_logs": len(counts["log_ids"]),
            "n_knot_groups": (
                counts["n_knot_groups"]
                if evaluation_domain
                == "gt_knot_group_spans"
                else ""
            ),
        })

    return sorted(
        volume_rows,
        key=lambda row: (
            row["evaluation_domain"],
            row["aggregate_scope"],
            row["dataset"],
            row["log_id"],
            row["size_category"],
            row["variant"],
        ),
    )

# ---------------------------------------------------------------------------
# CSV and plotting
# ---------------------------------------------------------------------------

def write_csv(path, rows, fieldnames=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def plot_profile_group(
    normed_rows,
    group_key,
    out_file,
):
    dataset, log_id, region_nr = group_key

    rows = [
        row
        for row in normed_rows
        if (
            row["dataset"] == dataset
            and row["log_id"] == log_id
            and int(row["region_nr"])
            == int(region_nr)
        )
    ]

    if not rows:
        return

    fig, ax = plt.subplots()
    plotted = False

    for variant in PROFILE_PLOT_VARIANTS:
        variant_rows = [
            row
            for row in rows
            if (
                row["variant"] == variant
                and row["dice"] != ""
            )
        ]

        if not variant_rows:
            continue

        variant_rows = sorted(
            variant_rows,
            key=lambda row: int(row["point_idx"]),
        )

        x = [
            float(row["relative_idx"])
            for row in variant_rows
        ]
        y = [
            float(row["dice"])
            for row in variant_rows
        ]

        style = PLOT_STYLE.get(
            variant,
            {
                "color": "black",
                "linestyle": "solid",
            },
        )

        ax.plot(
            x,
            y,
            label=VARIANT_LABELS.get(
                variant,
                variant,
            ),
            color=style["color"],
            linestyle=style["linestyle"],
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel(
        "Normed distance along knot group"
    )
    ax.set_ylabel("Dice")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(
        True,
        which="major",
        axis="both",
        linestyle=":",
        linewidth=0.6,
        alpha=0.5,
    )
    ax.set_axisbelow(True)
    ax.set_title(
        f"{log_id}, region {region_nr}"
    )
    ax.legend()

    os.makedirs(
        os.path.dirname(out_file),
        exist_ok=True,
    )
    plt.savefig(
        out_file,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)


def plot_all_region_profiles(
    normed_rows,
    root_out,
):
    groups = sorted({
        (
            row["dataset"],
            row["log_id"],
            int(row["region_nr"]),
        )
        for row in normed_rows
    })

    for dataset, log_id, region_nr in groups:
        out_file = os.path.join(
            root_out,
            "profiles",
            "plots",
            "per_knot",
            dataset,
            log_id,
            (
                f"{log_id}_region_"
                f"{region_nr:02d}_dice_profile.pdf"
            ),
        )

        plot_profile_group(
            normed_rows,
            (
                dataset,
                log_id,
                region_nr,
            ),
            out_file,
        )

def aggregate_normed_profiles(normed_rows):
    """Aggregate profiles by log, dataset, and knot-size class."""
    grouped = {}

    for row in normed_rows:
        if row["dice"] == "":
            continue

        size_groups = [
            "all",
            row["size_category"],
        ]
        scopes = [
            (
                "log",
                row["dataset"],
                row["log_id"],
            ),
            (
                "all_logs",
                row["dataset"],
                "ALL",
            ),
        ]

        for aggregate_scope, dataset, log_id in scopes:
            for size_category in size_groups:
                key = (
                    aggregate_scope,
                    dataset,
                    log_id,
                    size_category,
                    row["variant"],
                    int(row["point_idx"]),
                    float(row["relative_idx"]),
                )

                grouped.setdefault(key, []).append(
                    float(row["dice"])
                )

    aggregate_rows = []

    for key, values in grouped.items():
        (
            aggregate_scope,
            dataset,
            log_id,
            size_category,
            variant,
            point_idx,
            relative_idx,
        ) = key

        values_array = np.asarray(
            values,
            dtype=np.float32,
        )

        aggregate_rows.append({
            "aggregate_scope": aggregate_scope,
            "dataset": dataset,
            "log_id": log_id,
            "size_category": size_category,
            "variant": variant,
            "point_idx": point_idx,
            "relative_idx": safe_float(relative_idx),
            "mean_dice": safe_float(
                float(np.nanmean(values_array))
            ),
            "std_dice": safe_float(
                float(np.nanstd(values_array))
            ),
            "n_knot_groups": int(
                np.sum(~np.isnan(values_array))
            ),
        })

    return sorted(
        aggregate_rows,
        key=lambda row: (
            row["aggregate_scope"],
            row["dataset"],
            row["log_id"],
            row["size_category"],
            row["variant"],
            int(row["point_idx"]),
        ),
    )

def plot_aggregate_profile(
    aggregate_rows,
    aggregate_scope,
    dataset,
    log_id,
    size_category,
    stat,
    out_file,
):
    rows = [
        row
        for row in aggregate_rows
        if (
            row["aggregate_scope"]
            == aggregate_scope
            and row["dataset"] == dataset
            and row["log_id"] == log_id
            and row["size_category"]
            == size_category
        )
    ]

    if not rows:
        return

    value_key = f"{stat}_dice"
    fig, ax = plt.subplots()
    plotted = False

    for variant in PROFILE_PLOT_VARIANTS:
        variant_rows = [
            row
            for row in rows
            if (
                row["variant"] == variant
                and row[value_key] != ""
            )
        ]

        if not variant_rows:
            continue

        variant_rows = sorted(
            variant_rows,
            key=lambda row: int(row["point_idx"]),
        )

        x = [
            float(row["relative_idx"])
            for row in variant_rows
        ]
        y = [
            float(row[value_key])
            for row in variant_rows
        ]

        style = PLOT_STYLE.get(
            variant,
            {
                "color": "black",
                "linestyle": "solid",
            },
        )

        ax.plot(
            x,
            y,
            label=VARIANT_LABELS.get(
                variant,
                variant,
            ),
            color=style["color"],
            linestyle=style["linestyle"],
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel(
        "Normed distance along knot group"
    )
    ax.set_ylabel(
        f"{stat.capitalize()} Dice"
    )

    if stat == "mean":
        ax.set_ylim(-0.02, 1.02)
    
    ax.grid(
        True,
        which="major",
        axis="both",
        linestyle=":",
        linewidth=0.6,
        alpha=0.5,
    )
    ax.set_axisbelow(True)

    title_log = (
        "all logs"
        if log_id == "ALL"
        else f"log {log_id}"
    )
    title_size = (
        "all knot sizes"
        if size_category == "all"
        else f"{size_category} knots"
    )

    ax.set_title(
        f"{dataset}, {title_log}, {title_size}"
    )
    ax.legend()

    os.makedirs(
        os.path.dirname(out_file),
        exist_ok=True,
    )
    plt.savefig(
        out_file,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)


def plot_aggregate_mean_plus_sd(
    aggregate_rows,
    aggregate_scope,
    dataset,
    log_id,
    size_category,
    out_file,
):
    rows = [
        row
        for row in aggregate_rows
        if (
            row["aggregate_scope"]
            == aggregate_scope
            and row["dataset"] == dataset
            and row["log_id"] == log_id
            and row["size_category"]
            == size_category
        )
    ]

    if not rows:
        return

    fig, ax = plt.subplots()
    plotted = False

    for variant in PROFILE_PLOT_VARIANTS:
        variant_rows = [
            row
            for row in rows
            if (
                row["variant"] == variant
                and row["mean_dice"] != ""
                and row["std_dice"] != ""
            )
        ]

        if not variant_rows:
            continue

        variant_rows = sorted(
            variant_rows,
            key=lambda row: int(row["point_idx"]),
        )

        x = np.asarray([
            float(row["relative_idx"])
            for row in variant_rows
        ])
        mean = np.asarray([
            float(row["mean_dice"])
            for row in variant_rows
        ])
        std = np.asarray([
            float(row["std_dice"])
            for row in variant_rows
        ])

        style = PLOT_STYLE.get(
            variant,
            {
                "color": "black",
                "linestyle": "solid",
            },
        )
        color = style["color"]

        ax.plot(
            x,
            mean,
            label=VARIANT_LABELS.get(
                variant,
                variant,
            ),
            color=color,
            linestyle=style["linestyle"],
        )
        # ax.fill_between(
        #     x,
        #     np.clip(mean - std, 0.0, 1.0),
        #     np.clip(mean + std, 0.0, 1.0),
        #     color=color,
        #     alpha=0.16,
        #     linewidth=0,
        # )
        # Show SD at selected positions rather than as a continuous shaded band.
        error_indices = np.linspace(
            0,
            len(x) - 1,
            NO_ERROR_BARS,
            dtype=int,
        )

        ax.errorbar(
            x[error_indices],
            mean[error_indices],
            yerr=std[error_indices],
            fmt="none",
            ecolor=color,
            elinewidth=0.8,
            capsize=2.5,
            capthick=0.8,
            alpha=0.9,
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel(
        "Normed distance along knot group"
    )
    ax.set_ylabel("Mean Dice +/- 1 SD")
    ax.set_ylim(-0.02, 1.02)
    
    ax.grid(
        True,
        which="major",
        axis="both",
        linestyle=":",
        linewidth=0.6,
        alpha=0.5,
    )
    ax.set_axisbelow(True)

    title_log = (
        "all logs"
        if log_id == "ALL"
        else f"log {log_id}"
    )
    title_size = (
        "all knot sizes"
        if size_category == "all"
        else f"{size_category} knots"
    )

    ax.set_title(
        f"{dataset}, {title_log}, {title_size}"
    )
    ax.legend()

    os.makedirs(
        os.path.dirname(out_file),
        exist_ok=True,
    )
    plt.savefig(
        out_file,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)


def plot_aggregate_profiles(
    aggregate_rows,
    root_out,
):
    groups = sorted({
        (
            row["aggregate_scope"],
            row["dataset"],
            row["log_id"],
            row["size_category"],
        )
        for row in aggregate_rows
    })

    for (
        aggregate_scope,
        dataset,
        log_id,
        size_category,
    ) in groups:
        if aggregate_scope == "all_logs":
            out_dir = os.path.join(
                root_out,
                "profiles",
                "aggregate",
                "all_logs",
                dataset,
                size_category,
            )
        else:
            out_dir = os.path.join(
                root_out,
                "profiles",
                "aggregate",
                "logs",
                dataset,
                log_id,
                size_category,
            )

        for stat in ["mean", "std"]:
            out_file = os.path.join(
                out_dir,
                (
                    f"{size_category.upper()}_{log_id}_aggregate_{stat}_dice_profile.pdf"
                ),
            )

            plot_aggregate_profile(
                aggregate_rows,
                aggregate_scope,
                dataset,
                log_id,
                size_category,
                stat,
                out_file,
            )

        out_file = os.path.join(
            out_dir,
            (
                f"{size_category.upper()}_{log_id}_aggregate_mean_plus_SD_dice_profile.pdf"
            ),
        )

        plot_aggregate_mean_plus_sd(
            aggregate_rows,
            aggregate_scope,
            dataset,
            log_id,
            size_category,
            out_file,
        )

def plot_knot_size_histograms(
    table_rows,
    threshold_rows,
    root_out,
):
    threshold_lookup = {
        row["dataset"]: row
        for row in threshold_rows
    }

    datasets = sorted({
        row["dataset"]
        for row in table_rows
    })

    for dataset in datasets:
        volumes = np.asarray(
            [
                int(row["gt_voxels"])
                for row in table_rows
                if row["dataset"] == dataset
            ],
            dtype=np.float64,
        )

        if len(volumes) == 0:
            continue

        thresholds = threshold_lookup[dataset]
        lower = float(
            thresholds[
                "lower_tercile_threshold_voxels"
            ]
        )
        upper = float(
            thresholds[
                "upper_tercile_threshold_voxels"
            ]
        )

        fig, ax = plt.subplots(
            figsize=(5.5, 3.8)
        )

        ax.hist(
            volumes,
            # bins="auto",
            bins=HIST_NO_BINS,
            edgecolor="black",
            linewidth=0.8,
        )
        ax.axvline(
            lower,
            color="red",
            linestyle="dashed",
            linewidth=1.0,
            label="33.3 percentile",
        )
        ax.axvline(
            upper,
            color="red",
            linestyle="dotted",
            linewidth=1.0,
            label="66.7 percentile",
        )

        ax.set_xlabel(
            "Knot-group volume [voxels]"
        )
        ax.set_ylabel(
            "Number of knot groups"
        )
        ax.set_title(
            f"{dataset}: knot-group size distribution"
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend()
        fig.tight_layout()

        out_file = os.path.join(
            root_out,
            "profiles",
            "size_analysis",
            dataset,
            "knot_group_volume_histogram.pdf",
        )

        os.makedirs(
            os.path.dirname(out_file),
            exist_ok=True,
        )
        fig.savefig(
            out_file,
            bbox_inches="tight",
        )
        plt.close(fig)

def contour_case_key(case):
    return case["dataset"], case["log_id"], int(case["region_nr"])


def select_best_worst_cases(table_rows, rank_by_variant, top_n):
    score_key = f"{rank_by_variant}_mean_dice"
    scored = [row for row in table_rows if row.get(score_key) not in ("", None)]
    scored = [row for row in scored if not math.isnan(float(row[score_key]))]
    scored = sorted(scored, key=lambda row: float(row[score_key]))

    worst = scored[:top_n]
    best = list(reversed(scored[-top_n:]))

    selected_rows = []
    selected_cases = []
    for category, rows in [("worst", worst), ("best", best)]:
        for rank, row in enumerate(rows, start=1):
            selected = dict(row)
            selected["selection"] = category
            selected["selection_rank"] = rank
            selected["ranking_variant"] = rank_by_variant
            selected["ranking_score"] = row[score_key]
            selected_rows.append(selected)
            selected_cases.append({
                "selection": category,
                "rank": rank,
                "ranking_variant": rank_by_variant,
                "dataset": row["dataset"],
                "log_id": row["log_id"],
                "region_nr": int(row["region_nr"]),
            })
    return selected_rows, selected_cases


def load_contour_labels(dataset, log_id, gt_label_dirs, ct_label_dirs, seg_variant_dirs):
    labels = {
        "GT": load_label_binary(find_label_file(gt_label_dirs[dataset], log_id)),
        "CT": load_label_binary(find_label_file(ct_label_dirs[dataset], log_id)),
    }
    for variant in SEG_VARIANTS:
        variant_dir = resolve_segmentation_dir(seg_variant_dirs, variant, dataset)
        labels[variant] = load_label_binary(find_segmentation_label_file(variant_dir, log_id))
    return labels


# def plot_contour_slice(ax, image, labels, plot_variants, idx, crop, log_id, region_nr, r_value):
#     xs, xe, ys, ye = crop
#     ax.imshow(crop_slice(image, xs, xe, ys, ye, idx), cmap="gray", alpha=1.0)
#     ax.text(
#         0.01,
#         0.01,
#         f"{log_id}, region {region_nr}, slice {idx}",
#         horizontalalignment="left",
#         verticalalignment="bottom",
#         transform=ax.transAxes,
#         color="white",
#     )
#     ax.text(
#         0.999,
#         0.01,
#         f"r = {round(r_value, 2)}",
#         horizontalalignment="right",
#         verticalalignment="bottom",
#         transform=ax.transAxes,
#         color="white",
#     )

#     legend_handles = []
#     legend_labels = []
#     for variant in plot_variants:
#         label_slice = crop_slice(labels[variant], xs, xe, ys, ye, idx)
#         if not np.any(label_slice):
#             continue
#         ax.contour(
#             label_slice,
#             levels=[0.5],
#             colors=[CONTOUR_COLORS[variant]],
#             linewidths=LINEWIDTH,
#             linestyles="solid",
#         )
#         legend_handles.append(plt.Line2D([], [], color=CONTOUR_COLORS[variant]))
#         legend_labels.append(VARIANT_LABELS.get(variant, variant))

#     ax.axis("off")
#     ax.legend(legend_handles, legend_labels, loc="upper right")

def plot_contour_slice(
    ax,
    image,
    labels,
    plot_variants,
    idx,
    crop,
    log_id,
    region_nr,
    r_value,
):
    x_start, x_stop, y_start, y_stop = crop

    ax.imshow(
        crop_slice(
            image,
            x_start,
            x_stop,
            y_start,
            y_stop,
            idx,
        ),
        cmap="gray",
        alpha=1.0,
    )

    ax.text(
        0.01,
        0.01,
        f"{log_id}, region {region_nr}, slice {idx}",
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=ax.transAxes,
        color="white",
    )
    ax.text(
        0.999,
        0.01,
        f"r = {round(r_value, 2)}",
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
        color="white",
    )

    legend_handles = []
    legend_labels = []

    for variant in plot_variants:
        # Always include every variant in the legend.
        legend_handles.append(
            plt.Line2D(
                [],
                [],
                color=CONTOUR_COLORS[variant],
                linewidth=LINEWIDTH,
                linestyle="solid",
            )
        )
        legend_labels.append(
            VARIANT_LABELS.get(
                variant,
                variant,
            )
        )

        label_slice = crop_slice(
            labels[variant],
            x_start,
            x_stop,
            y_start,
            y_stop,
            idx,
        )

        # Only draw the contour when it is present on this slice.
        if not np.any(label_slice):
            continue

        ax.contour(
            label_slice,
            levels=[0.5],
            colors=[CONTOUR_COLORS[variant]],
            linewidths=LINEWIDTH,
            linestyles="solid",
        )

    ax.axis("off")

    # if not legend_handles:
    #     return None

    return ax.legend(
        legend_handles,
        legend_labels,
        loc="upper right",
    )


def plot_contour_case(case, table_lookup, root_out, root_ct, gt_label_dirs, ct_label_dirs, seg_variant_dirs, contour_plot_groups):
    dataset = case["dataset"]
    log_id = case["log_id"]
    region_nr = int(case["region_nr"])
    selection = case.get("selection", "custom")
    ranking_variant = case.get("ranking_variant", "custom")
    rank = case.get("rank")
    key = (dataset, log_id, region_nr)
    if key not in table_lookup:
        print(f"Skipping contour plot for missing region {dataset}/{log_id}/region_{region_nr}")
        return

    row = table_lookup[key]
    slice_start = int(row["slice_start"])
    slice_end = int(row["slice_end"])
    denom = max(slice_end - slice_start, 1)

    image = load_volume(find_label_file(os.path.join(root_ct, dataset), log_id))
    crop = knot_group_crop(
        image,
        slice_start,
        slice_end,
        padding=KNOT_CROP_PAD,
    )
    labels = load_contour_labels(dataset, log_id, gt_label_dirs, ct_label_dirs, seg_variant_dirs)
    zmax = image.shape[2] - 1
    s_start = max(slice_start - SLICE_PAD, 0)
    s_end = min(slice_end + SLICE_PAD, zmax)

    for group_name, plot_variants in contour_plot_groups.items():
        missing = [variant for variant in plot_variants if variant not in labels]
        if missing:
            raise KeyError(f"Contour group '{group_name}' contains unknown variants: {missing}")

        rank_part = f"rank_{rank:02d}_" if rank is not None else ""
        out_dir = os.path.join(
            root_out,
            "visualisations",
            "contours",
            group_name,
            ranking_variant,
            selection,
            dataset,
            f"{rank_part}{log_id}_region_{region_nr:02d}",
        )
        os.makedirs(out_dir, exist_ok=True)
        
        for idx in range(s_start, s_end + 1):
            r_value = (idx - slice_start) / denom

            fig, ax = plt.subplots()

            legend = plot_contour_slice(
                ax,
                image,
                labels,
                plot_variants,
                idx,
                crop,
                log_id,
                region_nr,
                r_value,
            )

            filename = (
                f"{log_id}_region_"
                f"{region_nr:02d}_s{idx:03d}"
            )

            fig.savefig(
                os.path.join(
                    out_dir,
                    f"{filename}.png",
                ),
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,
            )

            if legend is not None:
                legend.remove()

            fig.savefig(
                os.path.join(
                    out_dir,
                    f"{filename}_naked.png",
                ),
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,
            )

            plt.close(fig)

        # for idx in range(s_start, s_end + 1):
        #     r_value = (idx - slice_start) / denom
        #     fig, ax = plt.subplots()
        #     plot_contour_slice(ax, image, labels, plot_variants, idx, crop, log_id, region_nr, r_value)
        #     plt.savefig(
        #         os.path.join(out_dir, f"{log_id}_region_{region_nr:02d}_s{idx:03d}.png"),
        #         dpi=300,
        #         bbox_inches="tight",
        #         pad_inches=0,
        #     )
        #     plt.close(fig)


def plot_selected_contours(cases, table_rows, root_out, root_ct, gt_label_dirs, ct_label_dirs, seg_variant_dirs, contour_plot_groups):
    table_lookup = {
        (row["dataset"], row["log_id"], int(row["region_nr"])): row
        for row in table_rows
    }

    unique_cases = []
    seen = set()
    for case in cases:
        key = (case.get("ranking_variant", "custom"), case.get("selection", "custom")) + contour_case_key(case)
        if key in seen:
            continue
        seen.add(key)
        unique_cases.append(case)

    for case in unique_cases:
        plot_contour_case(case, table_lookup, root_out, root_ct, gt_label_dirs, ct_label_dirs, seg_variant_dirs, contour_plot_groups)


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def main(): 
    valid_plot_variants = [
        "CT",
        *SEG_VARIANTS,
    ]
    unknown_plot_variants = [
        variant
        for variant in PROFILE_PLOT_VARIANTS
        if variant not in valid_plot_variants
    ]

    if unknown_plot_variants:
        raise ValueError(
            "Unknown PROFILE_PLOT_VARIANTS: "
            f"{unknown_plot_variants}"
        )
        
    gt_label_dirs = {
        "test": os.path.join(ROOT_CT, "test", "labels", "final"),
        "val": os.path.join(ROOT_CT, "val", "labels", "final"),
    }
    ct_label_dirs = {
        "test": os.path.join(ROOT_CT, "test", "labels", "UNET_LPDsample-CT"),
        "val": os.path.join(ROOT_CT, "val", "labels", "UNET_LPDsample-CT"),
    }

    os.makedirs(ROOT_OUT, exist_ok=True)
    config_path = os.path.join(ROOT_OUT, "config", "run_config.json")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as handle:
        json.dump({
            "root_ct": ROOT_CT,
            "root_seg_5": ROOT_SEG_5,
            "root_seg_10": ROOT_SEG_10,
            "root_seg_fullct": ROOT_SEG_FULLCT,
            "datasets": DATASETS,
            "profile_length": PROFILE_LENGTH,
            "reuse_profile_cache": REUSE_PROFILE_CACHE,
            "external_profile_cache_roots": EXTERNAL_PROFILE_CACHE_ROOTS,
            "rank_by_variants": RANK_BY_VARIANTS,
            "top_n": TOP_N,
            "max_workers": MAX_WORKERS,
            "contour_plot_groups": CONTOUR_PLOT_GROUPS,
            "extra_contour_groups": EXTRA_CONTOUR_GROUPS,
            "profile_plot_variants": PROFILE_PLOT_VARIANTS,
            "knot_regions": KNOT_REGIONS,
            "reuse_volume_metric_cache": REUSE_VOLUME_METRIC_CACHE,
            "volume_metric_cache_version": VOLUME_METRIC_CACHE_VERSION,
            "generate_size_histogram": GENERATE_SIZE_HISTOGRAM,
            "hist_no_bins": HIST_NO_BINS,
        }, handle, indent=2)

    tasks = []
    for dataset in DATASETS:
        for log_id in list_log_ids(gt_label_dirs[dataset]):
            tasks.append({
                "dataset": dataset,
                "log_id": log_id,
                "gt_dir": gt_label_dirs[dataset],
                "ct_dir": ct_label_dirs[dataset],
                "root_out": ROOT_OUT,
                "external_cache_roots": EXTERNAL_PROFILE_CACHE_ROOTS,
                "seg_variant_dirs": SEG_VARIANT_DIRS,
                "ordered_variants": ORDERED_VARIANTS,
                "seg_variants": SEG_VARIANTS,
                "profile_length": PROFILE_LENGTH,
            })

    table_rows = []
    raw_rows = []
    normed_rows = []
    log_volume_rows = []

    print(f"Processing {len(tasks)} logs with {MAX_WORKERS} workers...")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_log, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            print(f"  done {result['dataset']}/{result['log_id']}")
            table_rows.extend(result["table_rows"])
            raw_rows.extend(result["raw_rows"])
            normed_rows.extend(result["normed_rows"])
            log_volume_rows.extend(
                result["log_volume_rows"]
            )

        table_rows = sorted(
        table_rows,
        key=lambda row: (
            row["dataset"],
            row["log_id"],
            int(row["region_nr"]),
        ),
    )
    raw_rows = sorted(
        raw_rows,
        key=lambda row: (
            row["dataset"],
            row["log_id"],
            int(row["region_nr"]),
            row["variant"],
            int(row["slice_idx"]),
        ),
    )
    normed_rows = sorted(
        normed_rows,
        key=lambda row: (
            row["dataset"],
            row["log_id"],
            int(row["region_nr"]),
            row["variant"],
            int(row["point_idx"]),
        ),
    )
    log_volume_rows = sorted(
        log_volume_rows,
        key=lambda row: (
            row["dataset"],
            row["log_id"],
            row["variant"],
        ),
    )

    size_threshold_rows = assign_knot_size_categories(
        table_rows,
        raw_rows,
        normed_rows,
    )

    region_score_rows = (
        calculate_knot_region_scores(
            raw_rows,
            "raw",
            KNOT_REGIONS,
        )
        + calculate_knot_region_scores(
            normed_rows,
            "normed",
            KNOT_REGIONS,
        )
    )

    region_score_rows = sorted(
        region_score_rows,
        key=lambda row: (
            row["profile_type"],
            row["dataset"],
            row["log_id"],
            int(row["region_nr"]),
            row["variant"],
            row["knot_region"],
        ),
    )

    region_summary_rows = (
        aggregate_knot_region_scores(
            region_score_rows
        )
    )

    add_region_scores_to_knot_table(
        table_rows,
        region_score_rows,
    )

    aggregate_rows = aggregate_normed_profiles(
        normed_rows
    )

    evaluated_variants = [
        variant
        for variant in ORDERED_VARIANTS
        if variant != "GT"
    ]

    volume_summary_rows = aggregate_volume_dice(
        table_rows,
        log_volume_rows,
        evaluated_variants,
    )

    table_path = os.path.join(
        ROOT_OUT,
        "tables",
        "knot_group_performance.csv",
    )
    raw_profiles_path = os.path.join(
        ROOT_OUT,
        "profiles",
        "raw_dice_profiles_long.csv",
    )
    normed_profiles_path = os.path.join(
        ROOT_OUT,
        "profiles",
        "normed_dice_profiles_long.csv",
    )
    aggregate_profiles_path = os.path.join(
        ROOT_OUT,
        "profiles",
        "aggregate_dice_profiles_long.csv",
    )
    size_thresholds_path = os.path.join(
        ROOT_OUT,
        "tables",
        "size_analysis",
        "knot_size_tercile_thresholds.csv",
    )
    region_scores_path = os.path.join(
        ROOT_OUT,
        "tables",
        "region_analysis",
        "knot_region_scores_long.csv",
    )
    region_summary_path = os.path.join(
        ROOT_OUT,
        "tables",
        "region_analysis",
        "knot_region_summary_long.csv",
    )
    volume_summary_path = os.path.join(
        ROOT_OUT,
        "tables",
        "volume_analysis",
        "volume_dice_summary_long.csv",
    )

    write_csv(table_path, table_rows)
    write_csv(raw_profiles_path, raw_rows)
    write_csv(
        normed_profiles_path,
        normed_rows,
    )
    write_csv(
        aggregate_profiles_path,
        aggregate_rows,
    )
    write_csv(
        size_thresholds_path,
        size_threshold_rows,
    )
    write_csv(
        region_scores_path,
        region_score_rows,
    )
    write_csv(
        region_summary_path,
        region_summary_rows,
    )
    write_csv(
        volume_summary_path,
        volume_summary_rows,
    )

    extra_cases = [
        {
            "selection": "custom",
            "ranking_variant": "custom",
            "rank": None,
            **case,
        }
        for case in EXTRA_CONTOUR_GROUPS
    ]

    selected_cases = list(extra_cases)
    selected_paths = []

    if not CUSTOM_CONTOURS_ONLY:
        for rank_by_variant in RANK_BY_VARIANTS:
            selected_rows, variant_cases = (
                select_best_worst_cases(
                    table_rows,
                    rank_by_variant,
                    TOP_N,
                )
            )

            selected_path = os.path.join(
                ROOT_OUT,
                "tables",
                (
                    f"best_worst_top{TOP_N}_by_"
                    f"{rank_by_variant}.csv"
                ),
            )

            write_csv(
                selected_path,
                selected_rows,
            )

            selected_paths.append(selected_path)
            selected_cases.extend(
                variant_cases
            )
            
    # selected_cases = []
    # selected_paths = []
    # for rank_by_variant in RANK_BY_VARIANTS:
    #     selected_rows, variant_cases = select_best_worst_cases(table_rows, rank_by_variant, TOP_N)
    #     selected_path = os.path.join(ROOT_OUT, "tables", f"best_worst_top{TOP_N}_by_{rank_by_variant}.csv")
    #     write_csv(selected_path, selected_rows)
    #     selected_paths.append(selected_path)
    #     selected_cases.extend(variant_cases)

    # extra_cases = [
    #     {"selection": "custom", "ranking_variant": "custom", "rank": None, **case}
    #     for case in EXTRA_CONTOUR_GROUPS
    # ]
    # selected_cases.extend(extra_cases)
    
    if GENERATE_SIZE_HISTOGRAM:
        print("Writing knot-size histograms...")
        plot_knot_size_histograms(
            table_rows,
            size_threshold_rows,
            ROOT_OUT,
        )

    if GENERATE_PROFILE_PLOTS:
        print("Writing per-knot dice profile plots...")
        plot_all_region_profiles(normed_rows, ROOT_OUT)
        print("Writing aggregate dice profile plots...")
        plot_aggregate_profiles(aggregate_rows, ROOT_OUT)

    if GENERATE_CONTOURS:
        print("Writing contour overlays for selected best/worst/custom groups...")
        plot_selected_contours(
            selected_cases,
            table_rows,
            ROOT_OUT,
            ROOT_CT,
            gt_label_dirs,
            ct_label_dirs,
            SEG_VARIANT_DIRS,
            CONTOUR_PLOT_GROUPS,
        )

    print(f"Wrote knot group table: {table_path}")
    print(f"Wrote raw profiles: {raw_profiles_path}")
    print(f"Wrote normalized profiles: {normed_profiles_path}")
    print(f"Wrote aggregate profiles: {aggregate_profiles_path}")
    for selected_path in selected_paths:
        print(f"Wrote best/worst table: {selected_path}")
        print(
        f"Wrote size thresholds: "
        f"{size_thresholds_path}"
    )
    print(
        f"Wrote knot-region scores: "
        f"{region_scores_path}"
    )
    print(
        f"Wrote knot-region summaries: "
        f"{region_summary_path}"
    )
    print(
        f"Wrote volume-Dice summaries: "
        f"{volume_summary_path}"
    )


if __name__ == "__main__":
    main()
