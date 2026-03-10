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
from monai.transforms import AsDiscrete

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def find_label_file(label_dir, log_id):
    """Return the first file matching the log_id with .nrrd extension in label_dir."""
    files = glob.glob(os.path.join(label_dir, f"{log_id}*.nrrd"))
    if not files:
        raise FileNotFoundError(f"No '.nrrd' label file found for {log_id} in {label_dir}")
    return files[0]


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
                     root_lpd, root_out, lpd_variants, fbpunet_variants):
    """
    Compute the profiles_all dictionary by iterating through each dataset and log.
    Uses your saving scheme per log.
    Returns a dictionary profiles_all[dataset][log_id][variant][metric].
    """
    volumeMetrics_all = {d: {} for d in datasets}
    profiles_all = {d: {} for d in datasets}

    for dataset in datasets:
        print(f"Processing {dataset} set for profiles...")
        gt_dir = gt_label_dirs[dataset]
        ct_dir = ct_label_dirs[dataset]
        log_ids = sorted([f.split('.')[0] for f in os.listdir(gt_dir) if f.endswith('.nrrd')])

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

            # Load LPD variants
            for variant in lpd_variants:
                variant_dir = os.path.join(root_lpd, variant, dataset, "labels", f"UNET_{variant}")
                variant_file = find_label_file(variant_dir, log_id)
                labels[variant] = loadLabelBinary(variant_file)

            # Load FBP-UNet variants
            for variant in fbpunet_variants:
                variant_dir = os.path.join(root_lpd, variant, dataset, "labels", f"UNET_{variant}")
                variant_file = find_label_file(variant_dir, log_id)
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
            for log_id in sorted([f.split('.')[0] for f in os.listdir(gt_label_dirs[dataset]) if f.endswith('.nrrd')])}
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

            plot_profiles_LPD(mean_profiles_normed, "mean", metric, log_id, out_dir_log)
            plot_profiles_FBPUNET(mean_profiles_normed, "mean", metric, log_id, out_dir_log)

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
        plot_profiles_LPD(agg_profiles_mean, "mean", metric, f"{dataset}_ALL", out_dir_agg)
        plot_profiles_FBPUNET(agg_profiles_mean, "mean", metric, f"{dataset}_ALL", out_dir_agg)
        plot_profiles_LPD(agg_profiles_std, "std", metric, f"{dataset}_ALL", out_dir_agg)
        plot_profiles_FBPUNET(agg_profiles_std, "std", metric, f"{dataset}_ALL", out_dir_agg)
        
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


def summarize_slicedata(summary_results, ordered_variants, short_names):
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
    for i, name in enumerate(short_names):
        if "9 src" in name:
            column_tuples.append(("9 src. pos.", name))
        elif "5 src" in name:
            column_tuples.append(("5 src. pos.", name))
        else:
            column_tuples.append(("CT", ""))

    df.columns = pd.MultiIndex.from_tuples(column_tuples)
    

    # # super-category for columns are 9 src and 5 src:
    # df.columns = pd.MultiIndex.from_tuples(
    #     [("CT", "")] + 
    #     [("9 src. pos.", key) for key in neworder[1:4]] +
    #     [("5 src. pos.", key) for key in neworder[4:]]) 

    return df


def summarize_volumedata(summary_results, ordered_variants, short_names):
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
    for i, name in enumerate(short_names):
        if "9 src" in name:
            column_tuples.append(("9 src. pos.", name))
        elif "5 src" in name:
            column_tuples.append(("5 src. pos.", name))
        else:
            column_tuples.append(("CT", ""))
    
    df.columns = pd.MultiIndex.from_tuples(column_tuples)

    return df

# ---------------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------------

def main(metric="dice", reload=True):
    # Editable parameters and directory roots
    root_ct = "/media/Store-SSD/Stembank/pine/LPDsample/"
    root_lpd = "/media/Store-SSD/Stembank/pine-LPDrecon/nrrd/"
    root_out = "/home/aime/monai/Postprocessing/Profiles"

    # Define GT and CT label directories for test and val datasets
    gt_label_dirs = {"test": os.path.join(root_ct, "test", "labels", "final"),
                        "val": os.path.join(root_ct, "val", "labels", "final")}
    ct_label_dirs = {"test": os.path.join(root_ct, "test", "labels", "UNET_LPDsample-CT"),
                        "val": os.path.join(root_ct, "val", "labels", "UNET_LPDsample-CT")}

    # Obtain variant lists from root_lpd directory
    all_variants = os.listdir(root_lpd)

    # Define datasets, overall label variants, and metrics
    datasets = ['test', 'val']
    # label_variants = ['GT', 'CT'] + lpd_variants + fbpunet_variants
    metrics = ['dice', 'iou', 'hdf']

    # Define variant groups with desired ordering
    variant_groups = {
        "9src": {
            # "3D_UNET_reconstructions_9_src_pos": "3D UNet 9 src.",
            "2.5D_LPD_reconstructions_9_src_pos_3_cons_slices_last": "2.5D, 9 src. 3 sl. last",
            "2D_LPD_reconstructions_9_src_pos": "2D, 9 src.",
            "2D_UNET_reconstructions_9_src_pos": "2D UNet 9 src."
        },
        "5src": {
            # "3D_UNET_reconstructions_5_src_pos": "3D UNet 5 src.",
            "2.5D_LPD_reconstructions_5_src_pos_5_cons_slices_middle": "2.5D 5 src. 5 sl. mid",
            "2D_LPD_reconstructions_5_src_pos": "2D, 5 src.",
            "2D_UNET_reconstructions_5_src_pos": "2D UNet 5 src."
        },
        "baseline": {
            "CT": "CT"
        }
    }

    # Define display order for variant groups
    group_order = ["baseline", "9src", "5src"]

    # Extract ordered variants and their short names
    ordered_variants = ["GT"]
    short_names = []
    
    for group in group_order:
        for variant, short_name in variant_groups[group].items():
            ordered_variants.append(variant)
            short_names.append(short_name)

    lpd_variants = [v for v in ordered_variants if "LPD" in v]
    fbpunet_variants = [v for v in ordered_variants if "UNET" in v]


    # Define knot region boundaries as fractions (editable here)
    KNOT_REGIONS = {
        "Start": (0.0, 0.2),
        "Mid": (0.2, 0.8),
        "End": (0.8, 1.0),
        "Total": (0.0, 1.0)}
    

    # Option to reload profiles_all from disk to speed up runtime
    if reload:
        print("Reloading profiles from disk...")
        volumeMetrics_all, profiles_all = reload_profiles(root_out, gt_label_dirs, datasets, ordered_variants, metrics)
    else:
        volumeMetrics_all, profiles_all = compute_profiles(datasets, ordered_variants, metrics,
                                        gt_label_dirs, ct_label_dirs,
                                        root_lpd, root_out, lpd_variants, fbpunet_variants)

    # Perform knot region analysis (aggregating, plotting, and printing scores)
    summary_slices_normed, summary_results_raw = analyze_knot_regions(profiles_all, metric, root_out, KNOT_REGIONS)
    summary_volumes = analyze_volumeMetrics(volumeMetrics_all, metric)

     # Generate LaTeX table code from summary_results.
    if metric == "hdf":
        summary_slices_df = summarize_slicedata(summary_results_raw, ordered_variants, short_names)
    else:
        summary_slices_df = summarize_slicedata(summary_slices_normed, ordered_variants, short_names)
    print(summary_slices_df)
    summary_volumes_df = summarize_volumedata(summary_volumes, ordered_variants, short_names)
    print(summary_volumes_df)

    # Assume df_extended is already created using create_summary_dataframe_extended(summary_results)
    latex_table_slices = summary_slices_df.to_latex(multicolumn=True,
                                    multirow=True,
                                    caption=f"Mean {metric.capitalize()} scores for knot segmentation evaluated at different positions within knot groups on validation and test datasets. Results are compared across U-Net models trained on different reconstruction methods.",
                                    label=f"tab:unet-slicewise-{metric}",
                                    float_format="%.3g",
                                    bold_rows=True)
    
    latex_table_volumes = summary_volumes_df.to_latex(multicolumn=True,
                                    multirow=True,
                                    caption=f"Mean {metric.capitalize()} scores for knot segmentation evaluated at different positions within knot groups on validation and test datasets. Results are compared across U-Net models trained on different reconstruction methods.",
                                    label=f"tab:unet-volume-{metric}",
                                    float_format="%.3g",
                                    bold_rows=True)


    # save to tex file
    out_latex = os.path.join(root_out, f"summary_table_{metric}_regions.tex")
    with open(out_latex, "w") as f:
        f.write(latex_table_slices)

    out_latex = os.path.join(root_out, f"summary_table_{metric}_volumes.tex")
    with open(out_latex, "w") as f:
        f.write(latex_table_volumes)

# ---------------------------------------------------------------------------
# Run Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main(metric="dice", reload=True)
    # NB: need to reload for correct scores for HDF metric


# ---------------------------------------------------------------------------
# Experimental / Debug
# ---------------------------------------------------------------------------

# import pickle
# def compare_nested_dicts_bitwise(d1, d2):
#     """
#     Bitwise compare two nested dictionaries.
    
#     The function serializes each dictionary using pickle with the highest protocol,
#     then compares the resulting bytes. If every byte is identical, the dictionaries
#     are considered bitwise identical.
    
#     Note: This does not perform any special handling for NaNs or different but equivalent
#     representations. If any difference in the underlying binary representation exists,
#     the function returns False.
#     """
#     b1 = pickle.dumps(d1, protocol=pickle.HIGHEST_PROTOCOL)
#     b2 = pickle.dumps(d2, protocol=pickle.HIGHEST_PROTOCOL)
#     return b1 == b2


# dataset = "test"
# avg_dices = []
# full_dices = []
# avg_dices_nanto0 = []
# dummy_dices = []
# for log_id in test_ids:
#     gt_dir = gt_label_dirs[dataset]
#     ct_dir = ct_label_dirs[dataset]
#     gt_label = loadLabelBinary(find_label_file(gt_dir, log_id))
#     ct_label = loadLabelBinary(find_label_file(ct_dir, log_id))

#     dice_metric = DiceMetric(include_background=False, reduction="mean")

#     y_pred = ct_label
#     y = gt_label

#     dice_score = []
#     dummy_score = []

#     for i in range(y_pred.shape[2]):
#         slice_y_pred = y_pred[:, :, i].unsqueeze(0).unsqueeze(0)
#         slice_y = y[:, :, i].unsqueeze(0).unsqueeze(0)
#         dice_score.append(dice_metric(y_pred=slice_y_pred, y=slice_y))
#         dummy_score.append(dice_metric(y_pred=slice_y, y=slice_y))

#     # mean but ignore nan
#     avg_dice = np.nanmean(dice_score)
#     avg_dices.append(avg_dice)
#     dummy_dice = np.nanmean(dummy_score)
#     dummy_dices.append(dummy_dice)

#     # mean but set nans to 0
#     avg_dice_nanto0 = np.nanmean(np.nan_to_num(dice_score, nan=0))
#     avg_dices_nanto0.append(avg_dice_nanto0)


#     full_dice = dice_metric(y_pred=y_pred.unsqueeze(0).unsqueeze(0), y=y.unsqueeze(0).unsqueeze(0))
#     full_dices.append(full_dice)

# for i, log_id in enumerate(test_ids):
#     print(f"Log {log_id}:")
#     print(f"Full Dice: {full_dices[i]}")
#     print(f"Avg Dice: {avg_dices[i]}")
#     print(f"Avg Dice (nan to 0): {avg_dices_nanto0[i]}")
#     print(f"Dummy Dice: {dummy_dices[i]}")
#     print("")

# # mean over full and avg dices
# avg_dices = np.array(avg_dices)
# full_dices = np.array(full_dices)
# np.nanmean(avg_dices)
# np.nanmean(full_dices)

# The key is that the Dice score is a ratio metric that doesn’t average linearly.
# When you compute it slice‐by‐slice and then average the results, each slice
# (even if it has a very small segmented area) gets equal weight. In contrast,
# when you compute the Dice score on the full 3D volume, the intersection and 
# union are summed over all slices, effectively giving more weight to slices with 
# more data. This difference in weighting can lead to a higher overall Dice score 
# when computed for the whole volume compared to the arithmetic mean of per‐slice scores.
# ---------------------------------------------------------------------------