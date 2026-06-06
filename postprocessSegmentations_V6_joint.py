#!/usr/bin/env python
"""
Region-level dice analysis for CT, 5 source-position LPD segmentations, and
9 source-position LPD segmentations.

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
ROOT_SEG_9 = "/media/Store-SSD/Stembank/pine-LPDseg/9srcpos/"
ROOT_OUT = "/home/aime/monai/Postprocessing/LPDseg_V6_knot_groups"

DATASETS = ["test"]
PROFILE_LENGTH = 101

# Set this to False if source labels changed and profiles must be recomputed.
REUSE_PROFILE_CACHE = True

# Previous V5 output folders are checked before recomputing profiles.
EXTERNAL_PROFILE_CACHE_ROOTS = [
    "/home/aime/monai/Postprocessing/LPDseg_5srcpos_9srcpos_vs_CT",
    "/home/aime/monai/Postprocessing/LPDseg_5srcpos_vs_CT",
]

MAX_WORKERS = min(6, max(1, os.cpu_count() or 1))
TOP_N = 5
RANK_BY_VARIANTS = ["5src-LPD-jnt", "9src-LPD-jnt"]

GENERATE_PROFILE_PLOTS = True
GENERATE_CONTOURS = True
SLICE_PAD = 3
CROP = (35, 30, 30, 50)  # xs, xe, ys, ye, using the same style as V4/V5
LINEWIDTH = 1.0

# Add manually requested knot groups here. These are plotted in addition to
# the automatically selected best/worst groups.
EXTRA_CONTOUR_GROUPS = [
    # {"dataset": "test", "log_id": "002753", "region_nr": 5},
]

SEG_VARIANTS = ["5src-LPD-seq", "5src-LPD-jnt", "9src-LPD-seq", "9src-LPD-jnt"]
ORDERED_VARIANTS = ["GT", "CT"] + SEG_VARIANTS

SEG_VARIANT_DIRS = {
    "5src-LPD-seq": {"root": ROOT_SEG_5, "folder": "sequential"},
    "5src-LPD-jnt": {"root": ROOT_SEG_5, "folder": "joint"},
    "9src-LPD-seq": {"root": ROOT_SEG_9, "folder": "sequential"},
    "9src-LPD-jnt": {"root": ROOT_SEG_9, "folder": "joint"},
}

VARIANT_LABELS = {
    "GT": "GT",
    "CT": "CT",
    "5src-LPD-seq": "5 src. sequential",
    "5src-LPD-jnt": "5 src. joint",
    "9src-LPD-seq": "9 src. sequential",
    "9src-LPD-jnt": "9 src. joint",
}

# 5src uses distinct colors from 9src in the all-source-position comparison.
PLOT_STYLE = {
    "CT": {"color": "blue", "linestyle": "dashed"},
    "5src-LPD-seq": {"color": "orange", "linestyle": "solid"},
    "5src-LPD-jnt": {"color": "cyan", "linestyle": "solid"},
    "9src-LPD-seq": {"color": "green", "linestyle": "solid"},
    "9src-LPD-jnt": {"color": "red", "linestyle": "solid"},
}

CONTOUR_COLORS = {
    "GT": "black",
    "CT": (0.1, 0.1, 1.0, 1.0),
    "5src-LPD-seq": "orange",
    "5src-LPD-jnt": "cyan",
    "9src-LPD-seq": (0.1, 1.0, 0.1, 1.0),
    "9src-LPD-jnt": "red",
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


def crop_slice(volume, xs, xe, ys, ye, idx):
    x_stop = -xe if xe else None
    y_stop = -ye if ye else None
    return volume[xs:x_stop, ys:y_stop, idx]


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
    seg_variants = task["seg_variants"]
    profile_length = task["profile_length"]

    profiles = load_cached_profiles(root_out, external_cache_roots, dataset, log_id, ordered_variants)

    gt_label = load_label_binary(find_label_file(gt_dir, log_id))
    if "GT" not in profiles:
        profiles["GT"] = dice_profile(gt_label, gt_label)

    if "CT" not in profiles:
        ct_label = load_label_binary(find_label_file(ct_dir, log_id))
        profiles["CT"] = dice_profile(ct_label, gt_label)

    for variant in seg_variants:
        if variant in profiles:
            continue
        variant_dir = resolve_segmentation_dir(seg_variant_dirs, variant, dataset)
        variant_label = load_label_binary(find_segmentation_label_file(variant_dir, log_id))
        profiles[variant] = dice_profile(variant_label, gt_label)

    save_cached_profiles(root_out, dataset, log_id, profiles)

    gt_indicator = np.nan_to_num(profiles["GT"], nan=0.0) > 0
    regions = connected_regions_1d(gt_indicator)

    table_rows = []
    raw_rows = []
    normed_rows = []

    for region in regions:
        region_nr = region["region_nr"]
        slice_start = region["slice_start"]
        slice_end = region["slice_end"]
        region_slice = slice(slice_start, slice_end + 1)
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

        for variant in ordered_variants:
            if variant == "GT":
                continue
            raw_profile = np.asarray(profiles[variant][region_slice], dtype=np.float32)
            normed_profile = interp_profile(raw_profile, profile_length)
            stats = profile_stats(raw_profile)
            for stat_name, value in stats.items():
                table_row[f"{variant}_{stat_name}_dice"] = safe_float(value)

            for offset, value in enumerate(raw_profile):
                raw_rows.append({
                    "dataset": dataset,
                    "log_id": log_id,
                    "region_nr": region_nr,
                    "slice_idx": slice_start + offset,
                    "relative_idx": safe_float(offset / max(length_slices - 1, 1)),
                    "variant": variant,
                    "dice": safe_float(value),
                })

            for point_idx, value in enumerate(normed_profile):
                normed_rows.append({
                    "dataset": dataset,
                    "log_id": log_id,
                    "region_nr": region_nr,
                    "point_idx": point_idx,
                    "relative_idx": safe_float(point_idx / max(profile_length - 1, 1)),
                    "variant": variant,
                    "dice": safe_float(value),
                })

        table_rows.append(table_row)

    return {
        "dataset": dataset,
        "log_id": log_id,
        "table_rows": table_rows,
        "raw_rows": raw_rows,
        "normed_rows": normed_rows,
    }


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


def plot_profile_group(normed_rows, group_key, out_file):
    dataset, log_id, region_nr = group_key
    rows = [row for row in normed_rows if (
        row["dataset"] == dataset and row["log_id"] == log_id and int(row["region_nr"]) == int(region_nr)
    )]
    if not rows:
        return

    fig, ax = plt.subplots()
    for variant in ["CT"] + SEG_VARIANTS:
        variant_rows = [row for row in rows if row["variant"] == variant and row["dice"] != ""]
        if not variant_rows:
            continue
        variant_rows = sorted(variant_rows, key=lambda row: int(row["point_idx"]))
        x = [float(row["relative_idx"]) for row in variant_rows]
        y = [float(row["dice"]) for row in variant_rows]
        style = PLOT_STYLE.get(variant, {"color": "black", "linestyle": "solid"})
        ax.plot(
            x,
            y,
            label=VARIANT_LABELS.get(variant, variant),
            color=style["color"],
            linestyle=style["linestyle"],
        )

    ax.set_xlabel("Normed distance along knot group")
    ax.set_ylabel("Dice")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{log_id}, region {region_nr}")
    ax.legend()
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def plot_all_region_profiles(normed_rows, root_out):
    groups = sorted({
        (row["dataset"], row["log_id"], int(row["region_nr"]))
        for row in normed_rows
    })
    for dataset, log_id, region_nr in groups:
        out_file = os.path.join(
            root_out,
            "profiles",
            "plots",
            dataset,
            log_id,
            f"{log_id}_region_{region_nr:02d}_dice_profile.pdf",
        )
        plot_profile_group(normed_rows, (dataset, log_id, region_nr), out_file)


def aggregate_normed_profiles(normed_rows):
    """Aggregate normalized knot profiles per log and over all logs."""
    grouped = {}
    for row in normed_rows:
        if row["dice"] == "":
            continue
        dataset = row["dataset"]
        log_id = row["log_id"]
        variant = row["variant"]
        point_idx = int(row["point_idx"])
        relative_idx = float(row["relative_idx"])
        value = float(row["dice"])

        keys = [
            ("log", dataset, log_id, variant, point_idx, relative_idx),
            ("all_logs", dataset, "ALL", variant, point_idx, relative_idx),
        ]
        for key in keys:
            grouped.setdefault(key, []).append(value)

    aggregate_rows = []
    for key, values in grouped.items():
        aggregate_scope, dataset, log_id, variant, point_idx, relative_idx = key
        values_array = np.asarray(values, dtype=np.float32)
        aggregate_rows.append({
            "aggregate_scope": aggregate_scope,
            "dataset": dataset,
            "log_id": log_id,
            "variant": variant,
            "point_idx": point_idx,
            "relative_idx": safe_float(relative_idx),
            "mean_dice": safe_float(float(np.nanmean(values_array))),
            "std_dice": safe_float(float(np.nanstd(values_array))),
            "n_knot_groups": int(np.sum(~np.isnan(values_array))),
        })

    return sorted(aggregate_rows, key=lambda row: (
        row["aggregate_scope"],
        row["dataset"],
        row["log_id"],
        row["variant"],
        int(row["point_idx"]),
    ))


def plot_aggregate_profile(aggregate_rows, aggregate_scope, dataset, log_id, stat, out_file):
    rows = [row for row in aggregate_rows if (
        row["aggregate_scope"] == aggregate_scope
        and row["dataset"] == dataset
        and row["log_id"] == log_id
    )]
    if not rows:
        return

    value_key = f"{stat}_dice"
    fig, ax = plt.subplots()
    for variant in ["CT"] + SEG_VARIANTS:
        variant_rows = [row for row in rows if row["variant"] == variant and row[value_key] != ""]
        if not variant_rows:
            continue
        variant_rows = sorted(variant_rows, key=lambda row: int(row["point_idx"]))
        x = [float(row["relative_idx"]) for row in variant_rows]
        y = [float(row[value_key]) for row in variant_rows]
        style = PLOT_STYLE.get(variant, {"color": "black", "linestyle": "solid"})
        ax.plot(
            x,
            y,
            label=VARIANT_LABELS.get(variant, variant),
            color=style["color"],
            linestyle=style["linestyle"],
        )

    ax.set_xlabel("Normed distance along knot group")
    ax.set_ylabel(f"{stat.capitalize()} Dice")
    if stat == "mean":
        ax.set_ylim(-0.02, 1.02)
    title_log = "all logs" if log_id == "ALL" else f"log {log_id}"
    ax.set_title(f"{dataset}, {title_log}")
    ax.legend()
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def plot_aggregate_profiles(aggregate_rows, root_out):
    groups = sorted({
        (row["aggregate_scope"], row["dataset"], row["log_id"])
        for row in aggregate_rows
    })
    for aggregate_scope, dataset, log_id in groups:
        group_dir = "all_logs" if aggregate_scope == "all_logs" else "logs"
        for stat in ["mean", "std"]:
            out_file = os.path.join(
                root_out,
                "profiles",
                "aggregate",
                group_dir,
                dataset,
                f"{log_id}_aggregate_{stat}_dice_profile.pdf",
            )
            plot_aggregate_profile(aggregate_rows, aggregate_scope, dataset, log_id, stat, out_file)


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


def plot_contour_slice(ax, image, labels, idx, crop, log_id, region_nr, r_value):
    xs, xe, ys, ye = crop
    ax.imshow(crop_slice(image, xs, xe, ys, ye, idx), cmap="gray", alpha=1.0)
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
    for variant in ORDERED_VARIANTS:
        label_slice = crop_slice(labels[variant], xs, xe, ys, ye, idx)
        if not np.any(label_slice):
            continue
        ax.contour(
            label_slice,
            levels=[0.5],
            colors=[CONTOUR_COLORS[variant]],
            linewidths=LINEWIDTH,
            linestyles="solid",
        )
        legend_handles.append(plt.Line2D([], [], color=CONTOUR_COLORS[variant]))
        legend_labels.append(VARIANT_LABELS.get(variant, variant))

    ax.axis("off")
    ax.legend(legend_handles, legend_labels, loc="upper right")


def plot_contour_case(case, table_lookup, root_out, root_ct, gt_label_dirs, ct_label_dirs, seg_variant_dirs):
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
    labels = load_contour_labels(dataset, log_id, gt_label_dirs, ct_label_dirs, seg_variant_dirs)
    zmax = image.shape[2] - 1
    s_start = max(slice_start - SLICE_PAD, 0)
    s_end = min(slice_end + SLICE_PAD, zmax)

    rank_part = f"rank_{rank:02d}_" if rank is not None else ""
    out_dir = os.path.join(
        root_out,
        "visualisations",
        "contours",
        ranking_variant,
        selection,
        dataset,
        f"{rank_part}{log_id}_region_{region_nr:02d}",
    )
    os.makedirs(out_dir, exist_ok=True)

    for idx in range(s_start, s_end + 1):
        r_value = (idx - slice_start) / denom
        fig, ax = plt.subplots()
        plot_contour_slice(ax, image, labels, idx, CROP, log_id, region_nr, r_value)
        plt.savefig(
            os.path.join(out_dir, f"{log_id}_region_{region_nr:02d}_s{idx:03d}.png"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(fig)


def plot_selected_contours(cases, table_rows, root_out, root_ct, gt_label_dirs, ct_label_dirs, seg_variant_dirs):
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
        plot_contour_case(case, table_lookup, root_out, root_ct, gt_label_dirs, ct_label_dirs, seg_variant_dirs)


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def main():
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
            "root_seg_9": ROOT_SEG_9,
            "datasets": DATASETS,
            "profile_length": PROFILE_LENGTH,
            "reuse_profile_cache": REUSE_PROFILE_CACHE,
            "external_profile_cache_roots": EXTERNAL_PROFILE_CACHE_ROOTS,
            "rank_by_variants": RANK_BY_VARIANTS,
            "top_n": TOP_N,
            "max_workers": MAX_WORKERS,
            "extra_contour_groups": EXTRA_CONTOUR_GROUPS,
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

    print(f"Processing {len(tasks)} logs with {MAX_WORKERS} workers...")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_log, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            print(f"  done {result['dataset']}/{result['log_id']}")
            table_rows.extend(result["table_rows"])
            raw_rows.extend(result["raw_rows"])
            normed_rows.extend(result["normed_rows"])

    table_rows = sorted(table_rows, key=lambda row: (row["dataset"], row["log_id"], int(row["region_nr"])))
    raw_rows = sorted(raw_rows, key=lambda row: (
        row["dataset"], row["log_id"], int(row["region_nr"]), row["variant"], int(row["slice_idx"])
    ))
    normed_rows = sorted(normed_rows, key=lambda row: (
        row["dataset"], row["log_id"], int(row["region_nr"]), row["variant"], int(row["point_idx"])
    ))

    table_path = os.path.join(ROOT_OUT, "tables", "knot_group_performance.csv")
    raw_profiles_path = os.path.join(ROOT_OUT, "profiles", "raw_dice_profiles_long.csv")
    normed_profiles_path = os.path.join(ROOT_OUT, "profiles", "normed_dice_profiles_long.csv")
    aggregate_profiles_path = os.path.join(ROOT_OUT, "profiles", "aggregate_dice_profiles_long.csv")
    aggregate_rows = aggregate_normed_profiles(normed_rows)
    write_csv(table_path, table_rows)
    write_csv(raw_profiles_path, raw_rows)
    write_csv(normed_profiles_path, normed_rows)
    write_csv(aggregate_profiles_path, aggregate_rows)

    selected_cases = []
    selected_paths = []
    for rank_by_variant in RANK_BY_VARIANTS:
        selected_rows, variant_cases = select_best_worst_cases(table_rows, rank_by_variant, TOP_N)
        selected_path = os.path.join(ROOT_OUT, "tables", f"best_worst_top{TOP_N}_by_{rank_by_variant}.csv")
        write_csv(selected_path, selected_rows)
        selected_paths.append(selected_path)
        selected_cases.extend(variant_cases)

    extra_cases = [
        {"selection": "custom", "ranking_variant": "custom", "rank": None, **case}
        for case in EXTRA_CONTOUR_GROUPS
    ]
    selected_cases.extend(extra_cases)

    if GENERATE_PROFILE_PLOTS:
        print("Writing per-knot dice profile plots...")
        plot_all_region_profiles(normed_rows, ROOT_OUT)
        print("Writing aggregate dice profile plots...")
        plot_aggregate_profiles(aggregate_rows, ROOT_OUT)

    if GENERATE_CONTOURS:
        print("Writing contour overlays for selected best/worst/custom groups...")
        plot_selected_contours(selected_cases, table_rows, ROOT_OUT, ROOT_CT, gt_label_dirs, ct_label_dirs, SEG_VARIANT_DIRS)

    print(f"Wrote knot group table: {table_path}")
    print(f"Wrote raw profiles: {raw_profiles_path}")
    print(f"Wrote normalized profiles: {normed_profiles_path}")
    print(f"Wrote aggregate profiles: {aggregate_profiles_path}")
    for selected_path in selected_paths:
        print(f"Wrote best/worst table: {selected_path}")


if __name__ == "__main__":
    main()
