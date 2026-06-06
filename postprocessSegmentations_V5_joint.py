#!/usr/bin/env python
"""
Script to evaluate 3D UNet segmentation results for knot volumes,
compute slice‐wise metrics and region‐based profiles, and produce
aggregated statistics.
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import nrrd
from scipy.ndimage import label
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def find_label_file(label_dir, log_id):
    """Return the first file matching log_id with .nrrd or .npy extension in label_dir."""
    patterns = [
        os.path.join(label_dir, f"{log_id}*.nrrd"),
        os.path.join(label_dir, f"{log_id}*.npy"),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No '.nrrd' or '.npy' label file found for {log_id} in {label_dir}")
    return files[0]


def find_segmentation_label_file(seg_variant_dir, log_id):
    """
    Return the segmentation label file for a log id.
    Segmentation outputs may use suffixes and may be stored as .nrrd or .npy,
    so we match by prefix and support both extensions.
    """
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


def resolve_segmentation_dir(seg_variant_dirs, variant, dataset):
    """
    Resolve the label directory for a segmentation variant.

    seg_variant_dirs entries may be either:
      - "joint"/"sequential" for the historical single-root layout, or
      - {"root": root_path, "folder": "joint"/"sequential"} for source-position layouts.
    """
    variant_config = seg_variant_dirs[variant]
    if isinstance(variant_config, dict):
        root_seg = variant_config["root"]
        variant_folder = variant_config["folder"]
    else:
        root_seg = seg_variant_dirs.get("_root", "")
        variant_folder = variant_config

    candidate_dirs = [
        os.path.join(root_seg, variant_folder, dataset, "labels"),
        os.path.join(root_seg, variant_folder, dataset),
        os.path.join(root_seg, variant_folder),
    ]
    variant_dir = next((path for path in candidate_dirs if os.path.isdir(path)), None)
    if variant_dir is None:
        raise FileNotFoundError(f"No segmentation directory found for '{variant}' in {candidate_dirs}")
    return variant_dir


def list_log_ids(label_dir):
    """List unique log ids in a label folder, supporting .nrrd and .npy."""
    names = []
    for fname in os.listdir(label_dir):
        if fname.endswith(".nrrd") or fname.endswith(".npy"):
            names.append(os.path.splitext(fname)[0])
    return sorted(set(names))


def loadLabelBinary(labelfile):
    """Load a label file (nrrd or npy) as a binary torch tensor."""
    if labelfile.endswith(".nrrd"):
        y, _ = nrrd.read(labelfile)
        y = np.array(y)
    elif labelfile.endswith(".npy"):
        y = np.load(labelfile)
        y = np.swapaxes(y, 0, 1)
    y[y > 0] = 1
    return torch.from_numpy(y)


def get3Dmetrics(y_pred, y):
    """
    Compute full 3D metrics for a given prediction and ground truth.
    Returns a dictionary with metric scores.
    """
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    iou_metric = MeanIoU(include_background=False, reduction="mean")
    hdf_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

    scores = {}
    scores['dice'] = dice_metric(y_pred=y_pred.unsqueeze(0).unsqueeze(0), y=y.unsqueeze(0).unsqueeze(0)).item()
    scores['iou'] = iou_metric(y_pred=y_pred.unsqueeze(0).unsqueeze(0), y=y.unsqueeze(0).unsqueeze(0)).item()
    scores['hdf'] = hdf_metric(y_pred=y_pred.unsqueeze(0).unsqueeze(0), y=y.unsqueeze(0).unsqueeze(0)).item()
    return scores


def getScoreProfiles(y_pred, y):
    """
    Calculate per-slice Dice, IoU, and Hausdorff scores.
    Returns three lists, one per metric.
    """
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    iou_metric = MeanIoU(include_background=False, reduction="mean")
    hdf_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    slice_dice_scores, slice_iou_scores, slice_hdf_scores = [], [], []

    for i in range(y_pred.shape[2]):
        slice_y_pred = y_pred[:, :, i].unsqueeze(0).unsqueeze(0)
        slice_y = y[:, :, i].unsqueeze(0).unsqueeze(0)
        slice_dice_scores.append(dice_metric(y_pred=slice_y_pred, y=slice_y).item())
        slice_iou_scores.append(iou_metric(y_pred=slice_y_pred, y=slice_y).item())
        slice_hdf_scores.append(hdf_metric(y_pred=slice_y_pred, y=slice_y).item())
    return slice_dice_scores, slice_iou_scores, slice_hdf_scores


def interp_profile(profile, newlength=101):
    """Interpolate a 1D profile to a fixed length."""
    oldlength = len(profile)
    return np.interp(np.linspace(0, 1, newlength),
                     np.linspace(0, 1, oldlength),
                     profile)


def get_knot_profiles(gt_indicator_profile, variant_profiles, newlength=101):
    """
    Extract raw and normalized knot profiles using connected components from GT.
    Returns two dicts: raw knotprofiles and normalized (interpolated) knotprofiles.
    """
    labeled_gt, num_regions = label(gt_indicator_profile > 0)
    knotprofiles_raw = {variant: [] for variant in variant_profiles}
    knotprofiles_normed = {variant: [] for variant in variant_profiles}

    for region in range(1, num_regions + 1):
        mask = labeled_gt == region
        if np.sum(mask) < 2:  # skip very small regions
            continue
        for variant, profile in variant_profiles.items():
            region_profile = profile[mask]
            knotprofiles_raw[variant].append(region_profile)
            knotprofiles_normed[variant].append(interp_profile(region_profile, newlength))
    return knotprofiles_raw, knotprofiles_normed


def aggregate_region_scores(profiles_dict, regions, metric):
    """
    Compute region scores for each variant.
    profiles_dict: dict { variant: list of 1D profiles }.
    regions: dict with region name keys and (start, end) fractions.
    Returns dict { variant: { region: [score, ...] } }.
    """
    region_scores = {variant: {r: [] for r in regions} for variant in profiles_dict}
    for variant, profile_list in profiles_dict.items():
        for profile in profile_list:
            for region, (start, end) in regions.items():
                idx_start = int(start * len(profile))
                idx_end = int(end * len(profile))
                prof_part = profile[idx_start:idx_end]
                if metric == "hdf":
                    prof_part = np.nan_to_num(prof_part, nan=0)
                    mask = prof_part != 0
                    prof_part = prof_part[mask]
                region_scores[variant][region].append(np.nanmean(prof_part))
    return region_scores


def getAbbreviation(variant):
    """
    Return a short label for a variant based on its name.
    Customize this function if needed.
    """
    source_position_labels = {
        "5src-LPD-seq": "5 src. LPD seg. sequential",
        "5src-LPD-jnt": "5 src. LPD seg. joint",
        "9src-LPD-seq": "9 src. LPD seg. sequential",
        "9src-LPD-jnt": "9 src. LPD seg. joint",
    }
    if variant in source_position_labels:
        return source_position_labels[variant]
    if variant == "LPD-seq":
        return "LPD seg. sequential"
    if variant == "LPD-jnt":
        return "LPD seg. joint"

    dim = variant.split("_")[0]
    if "LPD" in variant:
        pos = re.search(r'\d_src_pos', variant)[0][0] + " pos."
        sli_match = re.search(r'\d_cons_slices', variant)
        sli = sli_match[0][0] + " sl." if sli_match else ""
        met = variant.split("_")[-1]
        if met == "Mid":
            met = "mid"
        return " ".join(filter(None, [dim, pos, sli, met]))
    elif "UNET" in variant:
        pos = re.search(r'\d_src_pos', variant)[0][0] + " pos."
        return "FBP&" + dim + "-UNet " + pos
    else:
        return variant


def plot_profiles_LPD(normed_profiles, stat, metric, log_id, out_dir):
    """Plot profiles for LPD variants and CT."""
    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 101)
    if metric == "hdf":
        metricname = "95th percentile Hausdorff distance"
    else:
        metricname = metric.capitalize()

    variants = [v for v in normed_profiles.keys() if ("LPD" in v) or (v=="CT")]
    colors = ['blue', 'green', 'green', 'red', 'red']
    linestyles = ['dashed', 'solid', 'dotted', 'solid', 'dotted']

    for i, variant in enumerate(variants):
        label_str = getAbbreviation(variant)
        ax.plot(x, normed_profiles[variant], label=label_str,
                linestyle=linestyles[i % len(linestyles)],
                color=colors[i % len(colors)])
    ax.legend()
    ax.set_xlabel("Normed distance along knot group")
    ax.set_ylabel(f"{'Mean' if stat=='mean' else 'Std. Dev.'} {metricname}")

    os.makedirs(out_dir, exist_ok=True)
    outfile = os.path.join(out_dir, f"{log_id}_{stat}_{metric}_LPD.pdf")
    plt.savefig(outfile, bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_profiles_FBPUNET(normed_profiles, stat, metric, log_id, out_dir):
    """Plot profiles for FBP-Unet variants and CT."""
    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 101)
    if metric == "hdf":
        metricname = "95th percentile Hausdorff distance"
    else:
        metricname = metric.capitalize()

    variants = [v for v in normed_profiles.keys() if ("UNET" in v) or (v=="CT")]
    colors = ['blue', 'orange', 'purple']
    linestyles = ['dashed', 'solid', 'solid']

    for i, variant in enumerate(variants):
        label_str = getAbbreviation(variant)
        ax.plot(x, normed_profiles[variant], label=label_str,
                linestyle=linestyles[i % len(linestyles)],
                color=colors[i % len(colors)])
    ax.legend()
    ax.set_xlabel("Normed distance along knot group")
    ax.set_ylabel(f"{'Mean' if stat=='mean' else 'Std. Dev.'} {metricname}")

    os.makedirs(out_dir, exist_ok=True)
    outfile = os.path.join(out_dir, f"{log_id}_{stat}_{metric}_FBPUNET.pdf")
    plt.savefig(outfile, bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_profiles_segmentation(normed_profiles, stat, metric, log_id, out_dir):
    """Plot profiles for segmentation-specific variants and CT."""
    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 101)
    if metric == "hdf":
        metricname = "95th percentile Hausdorff distance"
    else:
        metricname = metric.capitalize()

    preferred_order = ["CT", "5src-LPD-seq", "5src-LPD-jnt", "9src-LPD-seq", "9src-LPD-jnt", "LPD-seq", "LPD-jnt"]
    variants = [v for v in preferred_order if v in normed_profiles]
    variants.extend([v for v in normed_profiles if v not in variants and v != "GT"])
    if not variants:
        plt.close()
        return

    style = {
        "CT": ("blue", "dashed"),
        "5src-LPD-seq": ("green", "solid"),
        "5src-LPD-jnt": ("red", "solid"),
        "9src-LPD-seq": ("green", "dotted"),
        "9src-LPD-jnt": ("red", "dotted"),
        "LPD-seq": ("green", "solid"),
        "LPD-jnt": ("red", "solid"),
    }
    for variant in variants:
        color, linestyle = style.get(variant, ("black", "solid"))
        ax.plot(x, normed_profiles[variant], label=getAbbreviation(variant), color=color, linestyle=linestyle)

    ax.legend()
    ax.set_xlabel("Normed distance along knot group")
    ax.set_ylabel(f"{'Mean' if stat=='mean' else 'Std. Dev.'} {metricname}")

    os.makedirs(out_dir, exist_ok=True)
    outfile = os.path.join(out_dir, f"{log_id}_{stat}_{metric}_SEG.pdf")
    plt.savefig(outfile, bbox_inches='tight', pad_inches=0)
    plt.close()


def load_volume_image(volume_file):
    """Load a CT volume (.nrrd or .npy)."""
    if volume_file.endswith(".nrrd"):
        img, _ = nrrd.read(volume_file)
        img = np.array(img)
    elif volume_file.endswith(".npy"):
        img = np.load(volume_file)
        img = np.swapaxes(img, 0, 1)
    else:
        raise ValueError(f"Unsupported volume format: {volume_file}")
    return img


def plotContourComparison(log_id, idx, r, xs, xe, ys, ye, plotvariants, colors, all_labels, ct_volume_file, linewidth=1, ax=None):
    """Old-style contour plotting adapted to current pipeline paths."""
    if ax is None:
        fig, ax = plt.subplots()

    # Show CT as background
    img = load_volume_image(ct_volume_file)
    imslice = img[xs:-xe, ys:-ye, idx]
    ax.imshow(imslice, cmap="gray", alpha=1.0)

    # write title in bottom right in white
    txt = log_id + ", slice " + str(idx)
    ax.text(.01, 0.01, txt, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, color='white')

    # write text on left bottom of image
    txt = "r = " + str(round(r, 2))
    ax.text(.999, 0.01, txt, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, color='white')

    # Plot the contours
    contours = {}

    # GT
    lblslice = all_labels['GT'][xs:-xe, ys:-ye, idx]
    contours["GT"] = ax.contour(lblslice, levels=[0.5], colors=[colors[0]], linewidths=linewidth, linestyles='solid')

    for j, variant in enumerate(plotvariants):
        lblslice = all_labels[variant][xs:-xe, ys:-ye, idx]
        contours[variant] = ax.contour(lblslice, levels=[0.5], colors=[colors[1 + j]], linewidths=linewidth, linestyles='solid')
    ax.axis('off')

    # Legend (same style as old script)
    legend_labels = [f'{label}' for label in contours]
    legend_handles = [plt.Line2D([], [], color=contour.collections[0].get_edgecolor()) for contour in contours.values()]
    ax.legend(legend_handles, legend_labels, loc='upper right')


def plot_critical_region_contours(
    profiles_all,
    root_ct,
    ct_label_dirs,
    gt_label_dirs,
    seg_variant_dirs,
    seg_variants,
    contour_cases,
    root_out,
):
    """
    Plot contour overlays for selected critical knot regions.
    Each case must contain:
      dataset, log_id, region_nr, slice_pad, crop=(xs, xe, ys, ye), linewidth
    """
    for case in contour_cases:
        dataset = case["dataset"]
        log_id = case["log_id"]
        region_nr = int(case.get("region_nr", 1))
        slice_pad = int(case.get("slice_pad", 3))
        crop = case.get("crop", (35, 30, 30, 50))
        linewidth = float(case.get("linewidth", 1.0))

        if dataset not in profiles_all or log_id not in profiles_all[dataset]:
            print(f"Skipping contour case; missing profiles for {dataset}/{log_id}")
            continue

        gt_profile = np.array(profiles_all[dataset][log_id]["GT"]["dice"]) > 0
        labeled_gt, nregions = label(gt_profile)
        if region_nr < 1 or region_nr > nregions:
            print(f"Skipping contour case; region {region_nr} not found in {dataset}/{log_id} (nregions={nregions})")
            continue

        ridx = np.where(labeled_gt == region_nr)[0]
        idxs, idxe = int(ridx[0]), int(ridx[-1])

        # Load CT image volume
        ct_img_file = find_label_file(os.path.join(root_ct, dataset), log_id)
        img = load_volume_image(ct_img_file)
        zmax = img.shape[2] - 1

        # Load contour labels exactly as old plotting expects
        labels = {
            "GT": loadLabelBinary(find_label_file(gt_label_dirs[dataset], log_id)).numpy(),
            "CT": loadLabelBinary(find_label_file(ct_label_dirs[dataset], log_id)).numpy(),
        }
        for variant in seg_variants:
            variant_dir = resolve_segmentation_dir(seg_variant_dirs, variant, dataset)
            labels[variant] = loadLabelBinary(find_segmentation_label_file(variant_dir, log_id)).numpy()

        s_start = max(idxs - slice_pad, 0)
        s_end = min(idxe + slice_pad, zmax)

        out_dir = os.path.join(root_out, dataset, "visualisations", "labelling", log_id, f"region_{region_nr}")
        denom = max(idxe - idxs, 1)
        os.makedirs(out_dir, exist_ok=True)

        # Show labels as contours instead of filled (exact old ordering)
        colors = ["black", "blue", "green", "red", "lime", "magenta"]
        colors[1] = (0.1, 0.1, 1.0, 1.0)
        colors[2] = (0.1, 1.0, 0.1, 1.0)

        varlist = []
        plotvariants = ["CT"] + list(seg_variants)
        varlist.append(plotvariants)

        xs, xe, ys, ye = crop
        for k, plotvariants in enumerate(varlist):
            for idx in range(s_start, s_end + 1):
                r = (idx - idxs) / denom
                plotContourComparison(
                    log_id, idx, r, xs, xe, ys, ye, plotvariants, colors, labels, ct_img_file, linewidth=linewidth
                )
                plt.savefig(
                    os.path.join(out_dir, f"{log_id}_labelling-{k+1}_s{idx}.png"),
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0
                )
                plt.close()


def save_profiles(dataset, log_id, variant_scores, metric, savedir):
    """Save computed profiles in npy and csv format (as before)."""
    out_path = os.path.join(savedir, dataset, metric)
    os.makedirs(out_path, exist_ok=True)
    save_dict = {variant: np.nan_to_num(scores[metric], nan=0)
                 for variant, scores in variant_scores.items()}
    np.save(os.path.join(out_path, f"{log_id}_{metric}_profiles.npy"), save_dict)
    df = pd.DataFrame(save_dict)
    df.to_csv(os.path.join(out_path, f"{log_id}_{metric}_profiles.csv"), index=False)

# ---------------------------------------------------------------------------
# Processing Functions
# ---------------------------------------------------------------------------

def compute_profiles(datasets, label_variants, metrics, gt_label_dirs, ct_label_dirs,
                     root_seg, root_out, seg_variants, seg_variant_dirs):
    """
    Compute the profiles_all dictionary by iterating through each dataset and log.
    Uses your saving scheme per log.
    Returns a dictionary profiles_all[dataset][log_id][variant][metric].
    """
    volumeMetrics_all = {d: {} for d in datasets}
    profiles_all = {d: {} for d in datasets}
    if root_seg is not None and "_root" not in seg_variant_dirs:
        seg_variant_dirs = {**seg_variant_dirs, "_root": root_seg}

    for dataset in datasets:
        print(f"Processing {dataset} set for profiles...")
        gt_dir = gt_label_dirs[dataset]
        ct_dir = ct_label_dirs[dataset]
        log_ids = list_log_ids(gt_dir)

        profiles_all[dataset] = {
            log_id: {
                variant: {
                    metric: [] for metric in metrics}
                for variant in label_variants}
            for log_id in log_ids}
        
        volumeMetrics_all[dataset] = {
            log_id: {
                variant: {}     # scroe types are computed in function
                for variant in label_variants}
            for log_id in log_ids}
        
        for log_id in log_ids:
            print(f"  Processing log {log_id}")
            gt_label = loadLabelBinary(find_label_file(gt_dir, log_id))
            ct_label = loadLabelBinary(find_label_file(ct_dir, log_id))
            labels = {"GT": gt_label, "CT": ct_label}

            # Load segmentation pipeline variants (sequential/joint)
            for variant in seg_variants:
                variant_dir = resolve_segmentation_dir(seg_variant_dirs, variant, dataset)
                variant_file = find_segmentation_label_file(variant_dir, log_id)
                labels[variant] = loadLabelBinary(variant_file)

            # Metrics
            for variant, pred_label in labels.items():
                # Compute slice-wise metrics
                dice_scores, iou_scores, hdf_scores = getScoreProfiles(pred_label, gt_label)
                profiles_all[dataset][log_id][variant]['dice'] = dice_scores
                profiles_all[dataset][log_id][variant]['iou'] = iou_scores
                profiles_all[dataset][log_id][variant]['hdf'] = hdf_scores
                # Compute full 3D metrics for each label
                full_scores = get3Dmetrics(pred_label, gt_label)
                volumeMetrics_all[dataset][log_id][variant] = full_scores
                
            # Save profiles per log
            for metric in metrics:
                save_profiles(dataset, log_id, profiles_all[dataset][log_id], metric, root_out)

            # Save full 3D metrics for all logs
            out_path = os.path.join(root_out, dataset)
            os.makedirs(out_path, exist_ok=True)
            np.save(os.path.join(out_path, f"{dataset}_volumeMetrics.npy"), volumeMetrics_all[dataset])
            # save csv
            df = pd.DataFrame(volumeMetrics_all[dataset])
            df.to_csv(os.path.join(out_path, f"{dataset}_volumeMetrics.csv"))
            
    return volumeMetrics_all, profiles_all


def reload_profiles(root_out, gt_label_dirs, datasets, label_variants, metrics):
    """
    Reload profiles from disk to speed up runtime.
    Returns profiles_all[dataset][log_id][variant][metric].
    """
    volumeMetrics_all = {d: {} for d in datasets}
    profiles_all = {d: {} for d in datasets}
    for dataset in datasets:
        #
        profiles_all[dataset] = {
            log_id: {
                variant: {
                    metric: np.load(os.path.join(root_out, dataset, metric, f"{log_id}_{metric}_profiles.npy"), allow_pickle=True)[()][variant]
                    for metric in metrics}
                for variant in label_variants}
            for log_id in list_log_ids(gt_label_dirs[dataset])}
        #
        volumeMetrics_all[dataset] = np.load(os.path.join(root_out, dataset, f"{dataset}_volumeMetrics.npy"), allow_pickle=True)[()]

    return volumeMetrics_all, profiles_all


def analyze_knot_regions(profiles_all, metric, root_out, KNOT_REGIONS):
    """
    Perform knot region analysis.
    For each dataset, extracts knot profiles from GT and variant dice profiles,
    plots individual and aggregated profiles, computes region scores, and prints mean scores.
    Returns a summary dictionary:
      summary_results[dataset][variant] = {"Start": value, "Mid": value, "End": value, "Total": value}
    """
    summary_results_normed = {}
    summary_results_raw = {}

    datasets = list(profiles_all.keys())
    log_ids = list(profiles_all[datasets[0]].keys())
    label_variants = list(profiles_all[datasets[0]][log_ids[0]].keys())

    for dataset in datasets:
        print(f"Aggregating knot region profiles for {dataset} set...")
        # Accumulators for normalized and raw knot profiles (skip GT)
        knotprofiles_normed_all = {variant: [] for variant in label_variants if variant != "GT"}
        knotprofiles_raw_all = {variant: [] for variant in label_variants if variant != "GT"}

        for log_id, variants_data in profiles_all[dataset].items():
            gt_indicator_profile = np.array(variants_data['GT']["dice"]) > 0    # binary GT indicator (is there knot or not?) -> must be DICE!!
            variant_profiles = {variant: np.array(variants_data[variant][metric])
                                for variant in label_variants if variant != "GT"}
            knotprofiles_raw, knotprofiles_normed = get_knot_profiles(gt_indicator_profile, variant_profiles)

            # Compute mean profiles per log for plotting (if available)
            mean_profiles_normed = {variant: np.mean(np.stack(profiles_list), axis=0)
                             for variant, profiles_list in knotprofiles_normed.items() if profiles_list}
            out_dir_log = os.path.join(root_out, dataset, "plots", "individual", metric)

            plot_profiles_segmentation(mean_profiles_normed, "mean", metric, log_id, out_dir_log)

            # Aggregate across logs
            for variant, profiles_list in knotprofiles_normed.items():
                knotprofiles_normed_all[variant].extend(profiles_list)
            for variant, profiles_list in knotprofiles_raw.items():
                knotprofiles_raw_all[variant].extend(profiles_list)

        # Aggregated profiles over all logs (only meaningful if not HDF since 0 or nans can destroy it)
        agg_profiles_mean = {}
        agg_profiles_std = {}
        for variant, profiles_list in knotprofiles_normed_all.items():
            if profiles_list:
                profiles_array = np.stack(profiles_list, axis=0)
                agg_profiles_mean[variant] = np.mean(profiles_array, axis=0)
                agg_profiles_std[variant] = np.std(profiles_array, axis=0)

        out_dir_agg = os.path.join(root_out, dataset, metric)
        plot_profiles_segmentation(agg_profiles_mean, "mean", metric, f"{dataset}_ALL", out_dir_agg)
        plot_profiles_segmentation(agg_profiles_std, "std", metric, f"{dataset}_ALL", out_dir_agg)
        
        # Compute region scores (raw and normalized)
        region_scores_normed = aggregate_region_scores(knotprofiles_normed_all, KNOT_REGIONS, metric)
        region_scores_raw = aggregate_region_scores(knotprofiles_raw_all, KNOT_REGIONS, metric)

        out_dir_scores = os.path.join(root_out, dataset, metric)
        pd.DataFrame({(v, r): scores for v, regs in region_scores_normed.items() for r, scores in regs.items()})\
            .to_csv(os.path.join(out_dir_scores, "region_scores_normed.csv"), index=False)
        pd.DataFrame({(v, r): scores for v, regs in region_scores_raw.items() for r, scores in regs.items()})\
            .to_csv(os.path.join(out_dir_scores, "region_scores.csv"), index=False)
        
        # Compute mean scores per region for normalized profiles
        # region_means_normed = {variant: {r: np.mean(scores) if scores else np.nan
        #                                  for r, scores in regs.items()}
        #                        for variant, regs in region_scores_normed.items()}
        # total_means_normed = {variant: np.mean(agg_profiles_mean[variant])
        #                       for variant in agg_profiles_mean.keys()}
        
        # print(f"\n[{dataset}] Normed Mean Scores per Variant:")
        # for variant in sorted(region_means_normed.keys()):
        #     print(f"{variant}: Start = {region_means_normed[variant]['Start']:.3f}, "
        #           f"Mid = {region_means_normed[variant]['Mid']:.3f}, "
        #           f"End = {region_means_normed[variant]['End']:.3f}, "
        #           f"Total = {total_means_normed.get(variant, np.nan):.3f}")
        
        # Similarly for raw profiles if needed
        # region_means_raw = {variant: {r: np.mean(scores) if scores else np.nan
        #                               for r, scores in regs.items()}
        #                     for variant, regs in region_scores_raw.items()}
        # total_means_raw = {}
        # for variant, profiles in knotprofiles_raw_all.items():
        #     means = [np.mean(profile) for profile in profiles if len(profile) > 0]
        #     total_means_raw[variant] = np.mean(means) if means else np.nan

        # print(f"\n[{dataset}] Raw Mean Scores per Variant:")
        # for variant in sorted(region_means_raw.keys()):
        #     print(f"{variant}: Start = {region_means_raw[variant]['Start']:.3f}, "
        #           f"Mid = {region_means_raw[variant]['Mid']:.3f}, "
        #           f"End = {region_means_raw[variant]['End']:.3f}, "
        #           f"Total = {total_means_raw.get(variant, np.nan):.3f}")

        # Build summary for the normed dataset
        dataset_summary = {}
        for variant, regs in region_scores_normed.items():
            # Calculate mean per region.
            variant_summary = {}
            for region in ["Start", "Mid", "End", "Total"]:
                variant_summary[region] = np.mean(regs[region]) if regs[region] else np.nan
            dataset_summary[variant] = variant_summary
        summary_results_normed[dataset] = dataset_summary

        # Build summary for the raw dataset
        dataset_summary = {}
        for variant, regs in region_scores_raw.items():
            # Calculate mean per region.
            variant_summary = {}
            for region in ["Start", "Mid", "End", "Total"]:
                variant_summary[region] = np.nanmean(regs[region]) if regs[region] else np.nan
            dataset_summary[variant] = variant_summary
        summary_results_raw[dataset] = dataset_summary
    
    return summary_results_normed, summary_results_raw


def analyze_volumeMetrics(volumeMetrics_all, metric):
    """
    Analyze volumeMetrics_all to compute mean scores for each variant.
    """
    datasets = list(volumeMetrics_all.keys())
    log_ids = list(volumeMetrics_all[datasets[0]].keys())
    variants = list(volumeMetrics_all[datasets[0]][log_ids[0]].keys())
    metrics = list(volumeMetrics_all[datasets[0]][log_ids[0]][variants[0]].keys())

    summary_results = {
        dataset: {
            variant: []
            for variant in variants}
        for dataset in datasets}
    
    # contraction over logs:
    for dataset in datasets:
        for log_id in volumeMetrics_all[dataset]:
            for variant in volumeMetrics_all[dataset][log_id]:
                summary_results[dataset][variant].append(volumeMetrics_all[dataset][log_id][variant][metric])

    # Compute mean over logs for each variant
    for dataset in datasets:
        for variant in summary_results[dataset]:
            summary_results[dataset][variant] = np.nanmean(summary_results[dataset][variant])
    
    return summary_results


def summarize_slicedata(summary_results, ordered_variants, short_names, short_groups):
    """
    Create a DataFrame with fixed columns.
    
    The fixed columns (in order) are:
       "CT", "5src_2D", "5src_2.5D", "9src_2D", "9src_2.5D".
       
    Rows are indexed by a MultiIndex with levels (Dataset, Position) where
    Position is one of ["Start", "Mid", "End", "Total"].
    """
    variants_pure = ordered_variants[1:]    # skip GT
    positions = ["Start", "Mid", "End", "Total"]

    
    rows = []
    for ds in summary_results:
        for pos in positions:
            row = {"Dataset": ds, "Position": pos}
            # for key in short_keys:
            for i, variant in enumerate(variants_pure):
                value = summary_results[ds][variant][pos]
                # Round to 3 significant digits if numeric
                if isinstance(value, (int, float)):
                    value = float(f"{value:.3g}")
                row[short_names[i]] = value
            rows.append(row)

    df = pd.DataFrame(rows)
    df.set_index(["Dataset", "Position"], inplace=True)

     # Create multi-level columns based on variant groups
    column_tuples = []
    for name, group in zip(short_names, short_groups):
        if group == "baseline":
            column_tuples.append(("CT", ""))
        elif group == "segmentation":
            column_tuples.append(("Segmentation", name))
        elif group == "9src":
            column_tuples.append(("9 src. pos.", name))
        elif group == "5src":
            column_tuples.append(("5 src. pos.", name))
        else:
            column_tuples.append((group, name))

    df.columns = pd.MultiIndex.from_tuples(column_tuples)
    

    # # super-category for columns are 9 src and 5 src:
    # df.columns = pd.MultiIndex.from_tuples(
    #     [("CT", "")] + 
    #     [("9 src. pos.", key) for key in neworder[1:4]] +
    #     [("5 src. pos.", key) for key in neworder[4:]]) 

    return df


def summarize_volumedata(summary_results, ordered_variants, short_names, short_groups):
    """
    Create a DataFrame with fixed columns.
    
    The fixed column order is the same as in summarize_slicedata.
    Rows are indexed by a MultiIndex with levels (Dataset) where
    """
    variants_pure = ordered_variants[1:]    # skip GT

    rows = []
    for ds in summary_results:
        row = {"Dataset": ds}
        for i, variant in enumerate(variants_pure):
            value = summary_results[ds][variant]
            # Round to 3 significant digits if numeric
            if isinstance(value, (int, float)):
                value = float(f"{value:.3g}")
            row[short_names[i]] = value
        rows.append(row)

    df = pd.DataFrame(rows)
    df.set_index(["Dataset"], inplace=True)

    # Create multi-level columns based on variant groups
    column_tuples = []
    for name, group in zip(short_names, short_groups):
        if group == "baseline":
            column_tuples.append(("CT", ""))
        elif group == "segmentation":
            column_tuples.append(("Segmentation", name))
        elif group == "9src":
            column_tuples.append(("9 src. pos.", name))
        elif group == "5src":
            column_tuples.append(("5 src. pos.", name))
        else:
            column_tuples.append((group, name))
    
    df.columns = pd.MultiIndex.from_tuples(column_tuples)

    return df

# ---------------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------------

def run_segmentation_analysis(
    analysis_name,
    analysis_title,
    root_ct,
    root_out,
    datasets,
    metrics,
    metric,
    reload,
    gt_label_dirs,
    ct_label_dirs,
    ordered_variants,
    short_names,
    short_groups,
    seg_variants,
    seg_variant_dirs,
    KNOT_REGIONS,
    generate_critical_contours,
    contour_cases,
):
    """Run one CT/LPD segmentation comparison and save outputs under root_out."""
    print(f"\n=== {analysis_title} ===")
    os.makedirs(root_out, exist_ok=True)

    if reload:
        print("Reloading profiles from disk...")
        volumeMetrics_all, profiles_all = reload_profiles(root_out, gt_label_dirs, datasets, ordered_variants, metrics)
    else:
        volumeMetrics_all, profiles_all = compute_profiles(
            datasets,
            ordered_variants,
            metrics,
            gt_label_dirs,
            ct_label_dirs,
            None,
            root_out,
            seg_variants,
            seg_variant_dirs,
        )

    summary_slices_normed, summary_results_raw = analyze_knot_regions(profiles_all, metric, root_out, KNOT_REGIONS)
    summary_volumes = analyze_volumeMetrics(volumeMetrics_all, metric)

    if metric == "hdf":
        summary_slices_df = summarize_slicedata(summary_results_raw, ordered_variants, short_names, short_groups)
    else:
        summary_slices_df = summarize_slicedata(summary_slices_normed, ordered_variants, short_names, short_groups)
    print(summary_slices_df)

    summary_volumes_df = summarize_volumedata(summary_volumes, ordered_variants, short_names, short_groups)
    print(summary_volumes_df)

    latex_table_slices = summary_slices_df.to_latex(
        multicolumn=True,
        multirow=True,
        caption=f"Mean {metric.capitalize()} scores for knot segmentation comparing {analysis_title}.",
        label=f"tab:{analysis_name}-slicewise-{metric}",
        float_format="%.3g",
        bold_rows=True,
    )

    latex_table_volumes = summary_volumes_df.to_latex(
        multicolumn=True,
        multirow=True,
        caption=f"Mean {metric.capitalize()} volume scores for knot segmentation comparing {analysis_title}.",
        label=f"tab:{analysis_name}-volume-{metric}",
        float_format="%.3g",
        bold_rows=True,
    )

    out_latex = os.path.join(root_out, f"summary_table_{analysis_name}_{metric}_regions.tex")
    with open(out_latex, "w") as f:
        f.write(latex_table_slices)

    out_latex = os.path.join(root_out, f"summary_table_{analysis_name}_{metric}_volumes.tex")
    with open(out_latex, "w") as f:
        f.write(latex_table_volumes)

    if generate_critical_contours:
        plot_critical_region_contours(
            profiles_all=profiles_all,
            root_ct=root_ct,
            ct_label_dirs=ct_label_dirs,
            gt_label_dirs=gt_label_dirs,
            seg_variant_dirs=seg_variant_dirs,
            seg_variants=seg_variants,
            contour_cases=contour_cases,
            root_out=root_out,
        )


def main(metric="dice", reload=False):
    # Editable parameters and directory roots
    root_ct = "/media/Store-SSD/Stembank/pine/LPDsample/"
    root_seg_5 = "/media/Store-SSD/Stembank/pine-LPDseg/5srcpos/"
    root_seg_9 = "/media/Store-SSD/Stembank/pine-LPDseg/9srcpos/"
    root_out_5 = "/home/aime/monai/Postprocessing/LPDseg_5srcpos_vs_CT"
    root_out_all = "/home/aime/monai/Postprocessing/LPDseg_5srcpos_9srcpos_vs_CT"

    # Define GT and CT label directories for test and val datasets
    gt_label_dirs = {"test": os.path.join(root_ct, "test", "labels", "final"),
                        "val": os.path.join(root_ct, "val", "labels", "final")}
    ct_label_dirs = {"test": os.path.join(root_ct, "test", "labels", "UNET_LPDsample-CT"),
                        "val": os.path.join(root_ct, "val", "labels", "UNET_LPDsample-CT")}

    # Define datasets, overall label variants, and metrics
    # datasets = ['test', 'val']
    datasets = ['test']
    metrics = ['dice', 'iou', 'hdf']

    analyses = [
        {
            "analysis_name": "5srcpos_vs_CT",
            "analysis_title": "CT baseline with 5 source-position sequential and joint LPD segmentations",
            "root_out": root_out_5,
            "ordered_variants": ["GT", "CT", "5src-LPD-seq", "5src-LPD-jnt"],
            "short_names": ["CT", "5src seq.", "5src joint"],
            "short_groups": ["baseline", "5src", "5src"],
            "seg_variants": ["5src-LPD-seq", "5src-LPD-jnt"],
            "seg_variant_dirs": {
                "5src-LPD-seq": {"root": root_seg_5, "folder": "sequential"},
                "5src-LPD-jnt": {"root": root_seg_5, "folder": "joint"},
            },
        },
        {
            "analysis_name": "5srcpos_9srcpos_vs_CT",
            "analysis_title": "CT baseline with 5 and 9 source-position sequential and joint LPD segmentations",
            "root_out": root_out_all,
            "ordered_variants": ["GT", "CT", "5src-LPD-seq", "5src-LPD-jnt", "9src-LPD-seq", "9src-LPD-jnt"],
            "short_names": ["CT", "5src seq.", "5src joint", "9src seq.", "9src joint"],
            "short_groups": ["baseline", "5src", "5src", "9src", "9src"],
            "seg_variants": ["5src-LPD-seq", "5src-LPD-jnt", "9src-LPD-seq", "9src-LPD-jnt"],
            "seg_variant_dirs": {
                "5src-LPD-seq": {"root": root_seg_5, "folder": "sequential"},
                "5src-LPD-jnt": {"root": root_seg_5, "folder": "joint"},
                "9src-LPD-seq": {"root": root_seg_9, "folder": "sequential"},
                "9src-LPD-jnt": {"root": root_seg_9, "folder": "joint"},
            },
        },
    ]

    # Critical contour plotting configuration.
    # region_nr is the connected component index along the GT knot profile for this log.
    generate_critical_contours = True
    contour_cases = [
        {
            "dataset": "test",
            "log_id": "002753",
            "region_nr": 5,
            "slice_pad": 3,
            "crop": (35, 30, 30, 50),
            "linewidth": 1.0,
        },
    ]


    # Define knot region boundaries as fractions (editable here)
    KNOT_REGIONS = {
        "Start": (0.0, 0.2),
        "Mid": (0.2, 0.8),
        "End": (0.8, 1.0),
        "Total": (0.0, 1.0)}
    

    for analysis in analyses:
        run_segmentation_analysis(
            root_ct=root_ct,
            datasets=datasets,
            metrics=metrics,
            metric=metric,
            reload=reload,
            gt_label_dirs=gt_label_dirs,
            ct_label_dirs=ct_label_dirs,
            KNOT_REGIONS=KNOT_REGIONS,
            generate_critical_contours=generate_critical_contours,
            contour_cases=contour_cases,
            **analysis,
        )

# ---------------------------------------------------------------------------
# Run Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main(metric="dice", reload=False)
    # NB: need to reload for correct scores for HDF metric
