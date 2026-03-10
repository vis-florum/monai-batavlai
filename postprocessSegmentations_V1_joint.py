from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric
from monai.transforms import AsDiscrete
import torch
import os
import sys
import nibabel as nib
import nrrd
import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
from scipy.ndimage import label


def loadLabelBinary(labelfile):
    # depending if file is npy or nrrd load
    if labelfile.endswith(".nrrd"):
        y, _ = nrrd.read(labelfile)
        y = np.array(y)
    elif labelfile.endswith(".npy"):
        y = np.load(labelfile)
        # Switch x and y axes (numpy!)
        y = np.swapaxes(y, 0, 1)

    y[y > 0] = 1
    y = torch.from_numpy(y)
    return y

def getScoreProfiles(y_pred,y):
    # Initialize the Metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    iou_metric = MeanIoU(include_background=False, reduction="mean")
    hdf_metric = HausdorffDistanceMetric(include_background=False, reduction="mean")

    slice_dice_scores = []
    slice_iou_scores = []
    slice_hdf_scores = []

    for i in range(y_pred.shape[2]):  # Iterate over each slice
        slice_y_pred = y_pred[:, :, i].unsqueeze(0).unsqueeze(0)
        slice_y = y[:, :, i].unsqueeze(0).unsqueeze(0)
        
        # Calculate Dice score for the slice
        dice_score = dice_metric(y_pred=slice_y_pred, y=slice_y)
        iou_score  = iou_metric(y_pred=slice_y_pred, y=slice_y)
        hdf_score  = hdf_metric(y_pred=slice_y_pred, y=slice_y)
        slice_dice_scores.append(dice_score.item())
        slice_iou_scores.append(iou_score.item())
        slice_hdf_scores.append(hdf_score.item())

    return slice_dice_scores, slice_iou_scores, slice_hdf_scores


##################################################################
# Make slice score for an individual log but for each slice in the 3D volume

ct_root = "/media/Store-SSD/Stembank/pine/LPDsample/"
gt_label_dir = os.path.join(ct_root, "test", "labels", "final")
ct_label_dir = os.path.join(ct_root, "test", "labels", "UNET_LPDsample-CT")

lpd_root = "/media/Store-SSD/Stembank/pine-LPDseg/"
lpd_seq_label_dir = os.path.join(lpd_root, "sequential")
lpd_jnt_label_dir = os.path.join(lpd_root, "joint")

test_ids = sorted([f for f in os.listdir(gt_label_dir) if f.endswith('.nrrd')])
test_ids = [f.split(".")[0] for f in test_ids]

# For each test sample, and each training variant:
all_dice_profiles = {id: {"GT": {}, "CT": {}, "LPD-seq": {}, "LPD-jnt": {}} for id in test_ids}
all_iou_profiles  = {id: {"GT": {}, "CT": {}, "LPD-seq": {}, "LPD-jnt": {}} for id in test_ids}
all_hdf_profiles  = {id: {"GT": {}, "CT": {}, "LPD-seq": {}, "LPD-jnt": {}} for id in test_ids}
all_labels        = {id: {"GT": {}, "CT": {}, "LPD-seq": {}, "LPD-jnt": {}} for id in test_ids}

for id in test_ids:
    # Load labels
    gt_label = loadLabelBinary(os.path.join(gt_label_dir, id + ".nrrd"))
    ct_label = loadLabelBinary(os.path.join(ct_label_dir, id + ".nrrd"))
    # find file that starts with "id" in the list for lpds
    lpd_seq_file = [f for f in os.listdir(lpd_seq_label_dir) if f.startswith(id)][0]
    lpd_jnt_file = [f for f in os.listdir(lpd_jnt_label_dir) if f.startswith(id)][0]
    lpd_seq_label = loadLabelBinary(os.path.join(lpd_seq_label_dir, lpd_seq_file))
    lpd_jnt_label = loadLabelBinary(os.path.join(lpd_jnt_label_dir, lpd_jnt_file))

    category = "GT"
    slice_dice_scores, slice_iou_scores, slice_hdf_scores = getScoreProfiles(gt_label,gt_label)
    all_dice_profiles[id][category] = slice_dice_scores
    all_iou_profiles[id][category]  = slice_iou_scores
    all_hdf_profiles[id][category]  = slice_hdf_scores
    all_labels[id][category] = gt_label

    category = "CT"
    slice_dice_scores, slice_iou_scores, slice_hdf_scores = getScoreProfiles(ct_label,gt_label)
    all_dice_profiles[id][category] = slice_dice_scores
    all_iou_profiles[id][category]  = slice_iou_scores
    all_hdf_profiles[id][category]  = slice_hdf_scores
    all_labels[id][category] = ct_label

    category = "LPD-seq"
    slice_dice_scores, slice_iou_scores, slice_hdf_scores = getScoreProfiles(lpd_seq_label,gt_label)
    all_dice_profiles[id][category] = slice_dice_scores
    all_iou_profiles[id][category]  = slice_iou_scores
    all_hdf_profiles[id][category]  = slice_hdf_scores
    all_labels[id][category] = lpd_seq_label

    category = "LPD-jnt"
    slice_dice_scores, slice_iou_scores, slice_hdf_scores = getScoreProfiles(lpd_jnt_label,gt_label)
    all_dice_profiles[id][category] = slice_dice_scores
    all_iou_profiles[id][category]  = slice_iou_scores
    all_hdf_profiles[id][category]  = slice_hdf_scores
    all_labels[id][category] = lpd_jnt_label


##################################################################
# Save dice profiles

# new saveProfiles function
def saveProfiles(id, all_profiles, savedir):
    GT = all_profiles[id]['GT']
    CT = all_profiles[id]['CT']
    LPDseq = all_profiles[id]['LPD-seq']
    LPDjnt = all_profiles[id]['LPD-jnt']
    GT = np.nan_to_num(GT, nan=0)
    CT = np.nan_to_num(CT, nan=0)
    LPDseq = np.nan_to_num(LPDseq, nan=0)
    LPDjnt = np.nan_to_num(LPDjnt, nan=0)

    # Save the profiles to a csv
    np.savetxt(os.path.join(savedir, id + "_Dice-profiles.csv"),
                np.array([GT, CT, LPDseq, LPDjnt]).T,
                    delimiter=",",
                    header="GT,CT,LPDseq,LPDjnt",
                    comments="")
    
# Save all dice profiles
outdir = "/home/aime/monai/LPDnew"
if not os.path.exists(outdir):
    os.makedirs(outdir)
for id in test_ids:
    saveProfiles(id, all_dice_profiles, outdir)



##################################################################
# Analyse dice profiles
nr_variants = 4
root_out = "/home/aime/monai/LPDnew"
outdir = os.path.join(root_out, "visualisations/profiles")
if not os.path.exists(outdir):
    os.makedirs(outdir)

def plotProfiles(normed_profiles, profiletype, id, outdir):
    fig, ax = plt.subplots()

    # ax.plot(gt, label="GT", color='black', linestyle='solid')
    ax.plot(normed_profiles[1, :], label="CT", color='blue', linestyle='dashed')
    ax.plot(normed_profiles[2, :], label="LPD seq", color='green', linestyle='solid')
    ax.plot(normed_profiles[3, :], label="LPD jnt", color='red', linestyle='solid')

    ax.legend()
    ax.set_xlabel("Normed distance r along knot group")
    if profiletype == "mean":
        ax.set_ylabel("Mean Dice across knot groups")
    elif profiletype == "std":
        ax.set_ylabel("Std. Dev. of Dice across knot groups")

    # Save the plot
    outfile = os.path.join(outdir, id + "_Dice-profiles_" + profiletype + ".pdf")
    plt.savefig(outfile, bbox_inches='tight', pad_inches=0)
    plt.close()



normed_profiles_all = np.zeros((nr_variants, 101, 0))

# Single profiles
for i,id in enumerate(test_ids):
    # load a dice profile
    csv_file = os.path.join(root_out, id + "_Dice-profiles.csv")
    df = pd.read_csv(csv_file)

    # find connected components in the GT and label them
    GT = df["GT"] > 0
    GT, nr_regions = label(GT)

    normed_profiles = np.zeros((nr_variants, 101, nr_regions))

    for region_nr in range(1, nr_regions+1):
        # extract region from the profile
        seg = GT == region_nr
        gt = df["GT"][seg]
        ct = df["CT"][seg]
        lpdseq = df["LPDseq"][seg]
        lpdjnt = df["LPDjnt"][seg]

        # interpolate in the profiles to make them the same length
        grouplength = len(gt)
        newlength = 101
        gt = np.interp(np.linspace(0, 1, newlength), np.linspace(0, 1, grouplength), gt)
        ct = np.interp(np.linspace(0, 1, newlength), np.linspace(0, 1, grouplength), ct)
        lpdseq = np.interp(np.linspace(0, 1, newlength), np.linspace(0, 1, grouplength), lpdseq)
        lpdjnt = np.interp(np.linspace(0, 1, newlength), np.linspace(0, 1, grouplength), lpdjnt)
        
        normed_profiles[0, :, region_nr-1] = gt
        normed_profiles[1, :, region_nr-1] = ct
        normed_profiles[2, :, region_nr-1] = lpdseq
        normed_profiles[3, :, region_nr-1] = lpdjnt

    # Accumulate all normed profiles
    normed_profiles_m = np.mean(normed_profiles, axis=2)
    normed_profiles_std = np.std(normed_profiles, axis=2)
    
    normed_profiles_all = np.concatenate((normed_profiles_all, normed_profiles), axis=2)

    plotProfiles(normed_profiles_m, "mean", id, outdir)
    plotProfiles(normed_profiles_std, "std", id, outdir)
    

# Accumulate all
normed_profiles_all.shape
normed_profiles_all_m = np.mean(normed_profiles_all, axis=2)
normed_profiles_all_std = np.std(normed_profiles_all, axis=2)

plotProfiles(normed_profiles_all_m, "mean", "ALL", outdir)
plotProfiles(normed_profiles_all_std, "std", "ALL", outdir)



##################################################################
# Plot the profiles for a single test sample
    
def plotProfileIdx(id, all_profiles, idxs, idxe, outdir, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    GT = all_profiles[id]['GT'][idxs:idxe]
    GT = np.nan_to_num(GT, nan=0)
    ax.plot(GT, label="GT", color='black', linestyle='solid')

    CT = all_profiles[id]['CT'][idxs:idxe]
    ax.plot(CT, label="CT", color='blue', linestyle='dashed')
    
    profile = all_profiles[id]['LPD-seq'][idxs:idxe]
    profile = np.nan_to_num(profile, nan=0)
    ax.plot(profile, label="LPD seq", color='green', linestyle='solid')
    
    profile = all_profiles[id]['LPD-jnt'][idxs:idxe]
    profile = np.nan_to_num(profile, nan=0)
    ax.plot(profile, label="LPD jnt", color='red', linestyle='solid')

    # Set labels and ticks
    ax.set_xlabel("Slice number")
    ax.set_ylabel("Dice score")
    num_ticks = 10
    tick_step = max((idxe - idxs) // num_ticks, 2)
    ax.set_xticks(np.arange(0, idxe-idxs, tick_step))
    ax.set_xticklabels(np.arange(idxs, idxe, tick_step))

    ax.legend(loc='lower center')
    
    # write title in bottom right
    txt = "log " + id
    ax.text(.01, 0.98, txt, horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes, color='black')#, fontsize='large')
    
    # Save the plot
    outfile = os.path.join(outdir, id + "_Dice-profiles_" + str(idxs)+"-"+str(idxe) + ".pdf")
    plt.savefig(outfile, bbox_inches='tight', pad_inches=0)
    plt.close()


id = "002554"
idxs = 1
idxe = img.shape[2]
#plotHausAndDice(id, idxs, idxe)
plotProfileIdx(id, all_dice_profiles, idxs, idxe, outdir)


idxs = 105
idxe = 215
plotProfileIdx(id, all_dice_profiles, idxs, idxe, outdir)


idxs = 148
idxe = 165
plotProfileIdx(id, all_dice_profiles, idxs, idxe, outdir)


id = "002753"
idxs = 125
idxe = 225
plotProfileIdx(id, all_dice_profiles, idxs, idxe, outdir)

idxs = 167
idxe = 184
plotProfileIdx(id, all_dice_profiles, idxs, idxe, outdir)



##################################################################
# Plot the corresponding labels in the image slices from the CT test samples

def plotContourComparison(id, idx, r, xs, xe, ys, ye, plotvariants, colors, linewidth=1, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
        
    # Show CT as background
    img, header = nrrd.read(os.path.join(ct_root, "test", id + ".nrrd"))
    imslice = img[xs:-xe, ys:-ye, idx]
    ax.imshow(imslice, cmap="gray", alpha=1.)

    # write title in bottom right in white
    txt = id + ", slice " + str(idx)
    ax.text(.01, 0.01, txt, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, color='white')
    # explanation of arguments of ax.text
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html
    
    # write text on left bottom of image
    txt = "r = " + str(round(r, 2))
    ax.text(.999, 0.01, txt, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, color='white')    

    # Plot the contours
    contours = {}

    # GT
    lblslice = all_labels[id]['GT'][xs:-xe, ys:-ye, idx]
    contours["GT"] = ax.contour(lblslice, levels=[0.5], colors=colors[0], linewidths=linewidth, linestyles='solid')

    for j,variant in enumerate(plotvariants):
        lblslice = all_labels[id][variant][xs:-xe, ys:-ye, idx]
        contours[variant] = ax.contour(lblslice, levels=[0.5], colors=colors[1+j], linewidths=linewidth, linestyles='solid')
    ax.axis('off')
    
    # Legend
    legend_labels = [f'{label}' for label in contours]
    legend_handles = [plt.Line2D([], [], color=contour.collections[0].get_edgecolor()) for contour in contours.values()]
    ax.legend(legend_handles, legend_labels, loc='upper right')


####
outdir = os.path.join(root_out, "visualisations/labelling")
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Show labels as contours instead of filled
colors = ["black", "blue", "green", "red"]
# slightly lighter blue
colors[1] = (0.1, 0.1, 1.0, 1.0)
# slightly lighter green
colors[2] = (0.1, 1.0, 0.1, 1.0)

# Find the region of interest
id = "002753"
df = pd.read_csv(os.path.join(root_out, id + "_Dice-profiles.csv"))
GT = df["GT"] > 0
GT, nr_regions = label(GT)
gidx = np.where(GT == 5)
idxs = gidx[0][0]
idxe = gidx[0][-1]

varlist = []
plotvariants = ["CT", "LPD-seq", "LPD-jnt"]
varlist.append(plotvariants)

for k, plotvariants in enumerate(varlist):
    for idx in range(idxs-3, idxe+3):
        r = (idx-idxs)/(idxe-idxs)
        plotContourComparison(id, idx, r, 35, 30, 30, 50, plotvariants, colors, linewidth=1)
        plt.savefig(os.path.join(outdir, id + "_labelling-" + str(k+1) + "_s" + str(idx) + ".png"),dpi=300,bbox_inches='tight', pad_inches=0)
        plt.close()
