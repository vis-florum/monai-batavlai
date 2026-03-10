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



def getScoreProfiles(y_pred_file,y_file):
    # Read the nrrd files
    y_pred, _ = nrrd.read(y_pred_file)
    y, _ = nrrd.read(y_file)
    y_pred = np.array(y_pred)
    y = np.array(y)

    # convert to binary masks
    y_pred[y_pred > 0] = 1
    y[y > 0] = 1

    y_pred = torch.from_numpy(y_pred)
    y = torch.from_numpy(y)


    # Initialize the Metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    iou_metric = MeanIoU(include_background=False, reduction="mean")
    hdf_metric = HausdorffDistanceMetric(include_background=False, reduction="mean")
    
    # Create the transform
    # num_classes = 2
    # one_hot_transform = AsDiscrete(to_onehot=num_classes)
    # binarize_transform = AsDiscrete(to_onehot=num_classes,threshold=1)


    slice_dice_scores = []
    slice_iou_scores = []
    slice_hdf_scores = []

    for i in range(y_pred.shape[2]):  # Iterate over each slice
        slice_y_pred = y_pred[:, :, i].unsqueeze(0).unsqueeze(0)
        slice_y = y[:, :, i].unsqueeze(0).unsqueeze(0)
        
        # convert to one-hot format
        #slice_y = one_hot_transform(slice_y)
        # binarize y_pred
        #slice_y_pred = binarize_transform(slice_y_pred)

        
        # Calculate Dice score for the slice
        dice_score = dice_metric(y_pred=slice_y_pred, y=slice_y)
        iou_score  = iou_metric(y_pred=slice_y_pred, y=slice_y)
        hdf_score  = hdf_metric(y_pred=slice_y_pred, y=slice_y)
        slice_dice_scores.append(dice_score.item())
        slice_iou_scores.append(iou_score.item())
        slice_hdf_scores.append(hdf_score.item())

    return slice_dice_scores, slice_iou_scores, slice_hdf_scores


def getShortName(lpdvariant):
    pt_src = re.compile(r'\d+_src_pos')
    pt_slices = re.compile(r'\d+_cons_slices')
    src_pos = pt_src.search(lpdvariant).group()
    if "2.5D" in lpdvariant:
        cons_slices = pt_slices.search(lpdvariant).group()
        return "2.5D " + src_pos[0] + "-pos " + cons_slices[0] + "-slices"
    elif "2D" in lpdvariant:
        return "2D " + src_pos[0] + "-pos"



# Make a slice score for an individual log but for each slice in the 3D volume

ct_root = "/media/Store-SSD/Stembank/pine/LPDsample/"
lpd_root = "/media/Store-SSD/Stembank/pine-LPDrecon/nrrd/"

pred_dir = os.path.join(ct_root, "test", "labels", "final")
test_ids = sorted([f for f in os.listdir(pred_dir) if f.endswith('.nrrd')])
test_ids = [f.split(".")[0] for f in test_ids]

# For each test sample, each variant of trainig/inference and each LPD variant if applicable
all_dice_profiles = {id: {"GT": {}, "CT/CT": {}, "CT/LPD": {}, "LPD/LPD": {}} for id in test_ids}
all_iou_profiles  = {id: {"GT": {}, "CT/CT": {}, "CT/LPD": {}, "LPD/LPD": {}} for id in test_ids}
all_hdf_profiles  = {id: {"GT": {}, "CT/CT": {}, "CT/LPD": {}, "LPD/LPD": {}} for id in test_ids}
all_labels        = {id: {"GT": {}, "CT/CT": {}, "CT/LPD": {}, "LPD/LPD": {}} for id in test_ids}

# CT RECONSTRUCTIONS
category = "CT/CT"
variant = "CT"
label_dir = os.path.join(ct_root, "test", "labels")
pred_name = "UNET_LPDsample-CT"
pred_dir = os.path.join(label_dir, pred_name)
pred_list = sorted([f for f in os.listdir(pred_dir) if f.endswith('.nrrd')])

for n in range(len(pred_list)):
    file = pred_list[n]
    id, _ = file.split(".")
    
    y_pred_file = os.path.join(pred_dir, file)
    y_file = os.path.join(label_dir, "final", file)

    slice_dice_scores, slice_iou_scores, slice_hdf_scores = getScoreProfiles(y_pred_file,y_file)
    
    all_dice_profiles[id][category][variant] = slice_dice_scores
    all_iou_profiles[id][category][variant]  = slice_iou_scores
    all_hdf_profiles[id][category][variant]  = slice_hdf_scores
    
    y_pred, _ = nrrd.read(y_pred_file)
    all_labels[id][category][variant] = y_pred


# LPD RECONSTRUCTIONS
done_final = False
for lpd_dir in os.listdir(lpd_root):
    label_dir = os.path.join(lpd_root, lpd_dir, "test", "labels")
    GT_dir = os.path.join(label_dir, "final")
    
    for pred_name in os.listdir(label_dir):
        if pred_name == "final":
            if done_final:
                continue
            else:
                category = "GT"
                variant = "GT"
                done_final = True
        elif pred_name.startswith("UNET"):
            trainedon = pred_name.split("_")[1]
            if "CT" in trainedon:
                category = "CT/LPD"
            else:
                category = "LPD/LPD"
            variant = getShortName(lpd_dir)
        else:
            continue    # if pred_name is neither "final" nor starts with "UNET..." then skip
        
        pred_dir = os.path.join(label_dir, pred_name)
        pred_list = sorted([f for f in os.listdir(pred_dir) if f.endswith('.nrrd')])

        for n in range(len(pred_list)):
            file = pred_list[n]
            id, _ = file.split(".")
            
            y_pred_file = os.path.join(pred_dir, file)
            y_file = os.path.join(GT_dir, file)

            slice_dice_scores, slice_iou_scores, slice_hdf_scores = getScoreProfiles(y_pred_file,y_file)
            
            all_dice_profiles[id][category][variant] = slice_dice_scores
            all_iou_profiles[id][category][variant]  = slice_iou_scores
            all_hdf_profiles[id][category][variant]  = slice_hdf_scores

            y_pred, _ = nrrd.read(y_pred_file)
            all_labels[id][category][variant] = y_pred


##################################################################
# Save dice profiles

def saveProfiles(id, all_profiles, savedir):
    GT = all_profiles[id]['GT']['GT']
    GT = np.nan_to_num(GT, nan=0)

    CT = all_profiles[id]['CT/CT']['CT']
    CT = np.nan_to_num(CT, nan=0)

    LPD2D5 = all_profiles[id]['LPD/LPD']["2D 5-pos"]
    LPD2D5 = np.nan_to_num(LPD2D5, nan=0)

    LPD2D9 = all_profiles[id]['LPD/LPD']["2D 9-pos"]
    LPD2D9 = np.nan_to_num(LPD2D9, nan=0)

    LPD25D5 = all_profiles[id]['LPD/LPD']["2.5D 5-pos 5-slices"]
    LPD25D5 = np.nan_to_num(LPD25D5, nan=0)

    LPD25D9 = all_profiles[id]['LPD/LPD']["2.5D 9-pos 3-slices"]
    LPD25D9 = np.nan_to_num(LPD25D9, nan=0)

    # Save the profiles to a csv
    np.savetxt(os.path.join(savedir, id + "_Dice-profiles.csv"),
                np.array([GT, CT, LPD2D5, LPD2D9, LPD25D5, LPD25D9]).T,
                    delimiter=",",
                    header="GT,CT,LPD2D5,LPD2D9,LPD25D5,LPD25D9",
                    comments="")


# Save all dice profiles
for id in test_ids:
    saveProfiles(id, all_dice_profiles, "/home/aime/monai/")


##################################################################
# Analyse dice profiles


outdir = "./visualisations/profiles"

normed_profiles_all = np.zeros((6, 101, 0))

for i,id in enumerate(test_ids):
    # load a dice profile
    df = pd.read_csv("/home/aime/monai/" + id + "_Dice-profiles.csv")

    # find connected components in the GT and label them
    GT = df["GT"] > 0
    GT, nregions = label(GT)

    normed_profiles = np.zeros((6, 101, nregions))

    for region_nr in range(1, nregions+1):
        seg = GT == region_nr    # for extracting from array
        gt = df["GT"][seg]
        ct = df["CT"][seg]
        lpd2d5 = df["LPD2D5"][seg]
        lpd2d9 = df["LPD2D9"][seg]
        lpd25d5 = df["LPD25D5"][seg]
        lpd25d9 = df["LPD25D9"][seg]

        # interpolate in the profiles to make them the same length
        grouplength = len(gt)
        newlength = 101
        gt = np.interp(np.linspace(0, 1, newlength), np.linspace(0, 1, grouplength), gt)
        ct = np.interp(np.linspace(0, 1, newlength), np.linspace(0, 1, grouplength), ct)
        lpd2d5 = np.interp(np.linspace(0, 1, newlength), np.linspace(0, 1, grouplength), lpd2d5)
        lpd25d9 = np.interp(np.linspace(0, 1, newlength), np.linspace(0, 1, grouplength), lpd25d9)
        lpd25d5 = np.interp(np.linspace(0, 1, newlength), np.linspace(0, 1, grouplength), lpd25d5)
        lpd2d9 = np.interp(np.linspace(0, 1, newlength), np.linspace(0, 1, grouplength), lpd2d9)

        normed_profiles[0, :, region_nr-1] = gt
        normed_profiles[1, :, region_nr-1] = ct
        normed_profiles[2, :, region_nr-1] = lpd2d5
        normed_profiles[3, :, region_nr-1] = lpd2d9
        normed_profiles[4, :, region_nr-1] = lpd25d5
        normed_profiles[5, :, region_nr-1] = lpd25d9

    # Accumulate all normed profiles
    normed_profiles_m = np.mean(normed_profiles, axis=2)
    normed_profiles_std = np.std(normed_profiles, axis=2)
    
    normed_profiles_all = np.concatenate((normed_profiles_all, normed_profiles), axis=2)


    # Plotting
    fig, ax = plt.subplots()

    variants = ["2.5D 9-pos 3-slices", "2D 9-pos", "2.5D 5-pos 5-slices", "2D 5-pos"]
    colors = ["green", "green", "red", "red"]
    linestyles = ["solid", "dotted", "solid", "dotted"]

    ax.plot(gt, label="GT", color='black', linestyle='solid')
    ax.plot(normed_profiles_m[1, :], label="CT", color='blue', linestyle='dashed')
    ax.plot(normed_profiles_m[5, :], label=variants[0], linestyle=linestyles[0], color=colors[0])
    ax.plot(normed_profiles_m[3, :], label=variants[1], linestyle=linestyles[1], color=colors[1])
    ax.plot(normed_profiles_m[4, :], label=variants[2], linestyle=linestyles[2], color=colors[2])
    ax.plot(normed_profiles_m[2, :], label=variants[3], linestyle=linestyles[3], color=colors[3])

    ax.legend()
    ax.set_xlabel("Normed distance along knot group")
    ax.set_ylabel("Mean Dice across knot groups")

    # Save the plot
    outfile = os.path.join(outdir, id + "_Dice-profiles_mean.pdf")
    plt.savefig(outfile, bbox_inches='tight', pad_inches=0)
    plt.close()

    # STD
    fig, ax = plt.subplots()

    variants = ["2.5D 9-pos 3-slices", "2D 9-pos", "2.5D 5-pos 5-slices", "2D 5-pos"]
    colors = ["green", "green", "red", "red"]
    linestyles = ["solid", "dotted", "solid", "dotted"]

    ax.plot(normed_profiles_std[1, :], label="CT", color='blue', linestyle='dashed')
    ax.plot(normed_profiles_std[5, :], label=variants[0], linestyle=linestyles[0], color=colors[0])
    ax.plot(normed_profiles_std[3, :], label=variants[1], linestyle=linestyles[1], color=colors[1])
    ax.plot(normed_profiles_std[4, :], label=variants[2], linestyle=linestyles[2], color=colors[2])
    ax.plot(normed_profiles_std[2, :], label=variants[3], linestyle=linestyles[3], color=colors[3])

    ax.legend()
    ax.set_xlabel("Normed distance along knot group")
    ax.set_ylabel("Standard Dev. of Dice across knot groups")

    # Save the plot
    outfile = os.path.join(outdir, id + "_Dice-profiles_std.pdf")
    plt.savefig(outfile, bbox_inches='tight', pad_inches=0)
    plt.close()

# Accumulate
normed_profiles_all.shape
normed_profiles_all_m = np.mean(normed_profiles_all, axis=2)
normed_profiles_all_std = np.std(normed_profiles_all, axis=2)

# Plotting
fig, ax = plt.subplots()

variants = ["2.5D 9-pos 3-slices", "2D 9-pos", "2.5D 5-pos 5-slices", "2D 5-pos"]
colors = ["green", "green", "red", "red"]
linestyles = ["solid", "dotted", "solid", "dotted"]

#ax.plot(gt, label="GT", color='black', linestyle='solid')
ax.plot(np.linspace(0, 1, 101), normed_profiles_all_m[1, :], label="CT", color='blue', linestyle='dashed')
ax.plot(np.linspace(0, 1, 101), normed_profiles_all_m[5, :], label=variants[0], linestyle=linestyles[0], color=colors[0])
ax.plot(np.linspace(0, 1, 101), normed_profiles_all_m[3, :], label=variants[1], linestyle=linestyles[1], color=colors[1])
ax.plot(np.linspace(0, 1, 101), normed_profiles_all_m[4, :], label=variants[2], linestyle=linestyles[2], color=colors[2])
ax.plot(np.linspace(0, 1, 101), normed_profiles_all_m[2, :], label=variants[3], linestyle=linestyles[3], color=colors[3])

ax.legend()
ax.set_xlabel("Normed distance r along knot group")
ax.set_ylabel("Mean Dice across knot groups")

# Save the plot
outfile = os.path.join(outdir, "ALL_Dice-profiles_mean.pdf")
plt.savefig(outfile, bbox_inches='tight', pad_inches=0)
plt.close()


fig, ax = plt.subplots()

variants = ["2.5D 9-pos 3-slices", "2D 9-pos", "2.5D 5-pos 5-slices", "2D 5-pos"]
colors = ["green", "green", "red", "red"]
linestyles = ["solid", "dotted", "solid", "dotted"]

ax.plot(np.linspace(0, 1, 101), normed_profiles_all_std[1, :], label="CT", color='blue', linestyle='dashed')
ax.plot(np.linspace(0, 1, 101), normed_profiles_all_std[5, :], label=variants[0], linestyle=linestyles[0], color=colors[0])
ax.plot(np.linspace(0, 1, 101), normed_profiles_all_std[3, :], label=variants[1], linestyle=linestyles[1], color=colors[1])
ax.plot(np.linspace(0, 1, 101), normed_profiles_all_std[4, :], label=variants[2], linestyle=linestyles[2], color=colors[2])
ax.plot(np.linspace(0, 1, 101), normed_profiles_all_std[2, :], label=variants[3], linestyle=linestyles[3], color=colors[3])

ax.legend()
ax.set_xlabel("Normed distance r along knot group")
ax.set_ylabel("Std. Dev. of Dice across knot groups")

# Save the plot
outfile = os.path.join(outdir, "ALL_Dice-profiles_std.pdf")
plt.savefig(outfile, bbox_inches='tight', pad_inches=0)
plt.close()



##################################################################
# Special plot for poster
outdir = "/home/aime/monai/visualisations/labelling/poster"
if not os.path.exists(outdir):
    os.makedirs(outdir)

id = "002753"
df = pd.read_csv("/home/aime/monai/" + id + "_Dice-profiles.csv")
# find connected components in the GT and label them
GT = df["GT"] > 0
GT, nregions = label(GT)
gidx = np.where(GT == 5)
idxs = gidx[0][0]
idxe = gidx[0][-1]

varlist = []
namelist = []
plotvariants = ["GT", "CT"]
varlist.append(plotvariants)
namelist.append("GT-CT")
plotvariants = ["2.5D 9-pos 3-slices"]
varlist.append(plotvariants)
namelist.append("LPD-2.5D9-3")
plotvariants = ["2.5D 5-pos 5-slices"]
varlist.append(plotvariants)
namelist.append("LPD-2.5D5-5")

xs = 35
xe = 30
ys = 30
ye = 50
colors = ["tab:red", "tab:cyan", "tab:purple", "turquoise", "red", "orange"]

for idx in [170, 175]:
    for k, plotvariants in enumerate(varlist):
        # Plot logic
        fig, ax = plt.subplots()

        img, header = nrrd.read(os.path.join(ct_root, "test", id + ".nrrd"))
        imslice = img[xs:-xe, ys:-ye, idx]

        ax.imshow(imslice, cmap="gray", alpha=1.)

        linewidth = 2
        for j,variant in enumerate(plotvariants):
            if "GT" in variant:
                category = "GT"
            elif "CT" in variant:
                category = "CT/CT"  
            else:
                category = "LPD/LPD"
            lblslice = all_labels[id][category][variant][xs:-xe, ys:-ye, idx]
            ax.contour(lblslice, levels=[0.5], colors=colors[j], linewidths=linewidth, linestyles='solid')

        ax.axis('off')

        plt.savefig(os.path.join(outdir, id + "_labelling_" + namelist[k] + "_s" + str(idx) + ".png"),dpi=300,bbox_inches='tight', pad_inches=0)
        plt.close()





##################################################################
def plot_dice_scores(id, all_profiles, idxs, idxe, npos, scoretype, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
        
    if npos == 5:
        variant1 = "2.5D 5-pos 5-slices"
        variant2 = "2D 5-pos"
    elif npos == 9:
        variant1 = "2.5D 9-pos 3-slices"
        variant2 = "2D 9-pos"

    # Plot logic
    ax.set_title(scoretype + " profiles, test sample " + id)

    GT = all_profiles[id]['GT']['GT'][idxs:idxe]
    GT = np.nan_to_num(GT, nan=0)
    ax.plot(GT, label="GT", color='black', linestyle='dotted')

    CT = all_profiles[id]['CT/CT']['CT'][idxs:idxe]
    ax.plot(CT, label="CT/CT", color='blue', linestyle='dashed')

    variant = variant1
    ax.plot(all_profiles[id]['CT/LPD'][variant][idxs:idxe], label="CT/LPD " + variant, color = 'turquoise')
    ax.plot(all_profiles[id]['LPD/LPD'][variant][idxs:idxe], label="LPD/LPD " + variant, color='red')

    variant = variant2
    ax.plot(all_profiles[id]['CT/LPD'][variant2][idxs:idxe], label="CT/LPD " + variant, color = 'green', linestyle='dashed')
    ax.plot(all_profiles[id]['LPD/LPD'][variant2][idxs:idxe], label="LPD/LPD " + variant, color='orange', linestyle='dashed')

    # Set labels and ticks
    ax.set_xlabel("Slice number")
    ax.set_ylabel(scoretype)
    num_ticks = 10
    tick_step = max((idxe - idxs) // num_ticks, 2)
    ax.set_xticks(np.arange(0, idxe-idxs, tick_step))
    ax.set_xticklabels(np.arange(idxs, idxe, tick_step))

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 1, 3, 5, 2, 4]
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='lower center')

    # If ax was None, show the plot
    if ax is None:
        plt.show()



def plotProfiles(id, all_profiles, idxs, idxe, scoretype, outdir):
    for npos in [5, 9]:
        plot_dice_scores(id, all_profiles, idxs, idxe, npos, scoretype)
        outfile = os.path.join(outdir, id + "_" + str(npos) + "-pos_" + scoretype + "-profile_" + str(idxs)+"-"+str(idxe) + ".pdf")
        plt.savefig(outfile)
        plt.close()

def plotHausAndDice(id, idxs, idxe):
    plotProfiles(id, all_hdf_profiles, idxs, idxe, "Hausdorff", outdir)
    plotProfiles(id, all_dice_profiles, idxs, idxe, "Dice", outdir)
    
    
def makeDiceProfile(id, all_profiles, idxs, idxe, outdir, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    # Plot logic
#    ax.set_title("Test set log " + id)

    GT = all_profiles[id]['GT']['GT'][idxs:idxe]
    GT = np.nan_to_num(GT, nan=0)
    ax.plot(GT, label="GT", color='black', linestyle='solid')

    CT = all_profiles[id]['CT/CT']['CT'][idxs:idxe]
    ax.plot(CT, label="CT", color='blue', linestyle='dashed')
    
    variants = ["2.5D 9-pos 3-slices", "2D 9-pos", "2.5D 5-pos 5-slices", "2D 5-pos"]
    colors = ["green", "green", "red", "red"]
    linestyles = ["solid", "dotted", "solid", "dotted"]
    
    for variant, color, linestyle in zip(variants, colors, linestyles):
        profile = all_profiles[id]['LPD/LPD'][variant][idxs:idxe]
        profile = np.nan_to_num(profile, nan=0)
        ax.plot(profile, label="LPD " + variant, color=color, linestyle=linestyle)

    # Set labels and ticks
    ax.set_xlabel("Slice number")
    ax.set_ylabel(scoretype)
    num_ticks = 10
    tick_step = max((idxe - idxs) // num_ticks, 2)
    ax.set_xticks(np.arange(0, idxe-idxs, tick_step))
    ax.set_xticklabels(np.arange(idxs, idxe, tick_step))

    # Legend
    #handles, labels = ax.get_legend_handles_labels()
    #order = [0, 1, 3, 5, 2, 4]
    #ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='lower center')
    ax.legend(loc='lower center')
    
    # write title in bottom right
    txt = "log " + id
    ax.text(.01, 0.98, txt, horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes, color='black')#, fontsize='large')
    

    # Save the plot
    outfile = os.path.join(outdir, id + "_Dice-profiles_" + str(idxs)+"-"+str(idxe) + ".pdf")
    plt.savefig(outfile, bbox_inches='tight', pad_inches=0)
    plt.close()




##################################################################
# Plot the profiles for a single test sample
visdir = "/home/aime/monai/visualisations/"
outdir = os.path.join(visdir, "profiles")



id = "002554"
idxs = 1
idxe = img.shape[2]
#plotHausAndDice(id, idxs, idxe)
makeDiceProfile(id, all_dice_profiles, idxs, idxe, outdir)


idxs = 105
idxe = 215
makeDiceProfile(id, all_dice_profiles, idxs, idxe, outdir)


idxs = 148
idxe = 165
makeDiceProfile(id, all_dice_profiles, idxs, idxe, outdir)


id = "002753"
idxs = 125
idxe = 225
makeDiceProfile(id, all_dice_profiles, idxs, idxe, outdir)

idxs = 167
idxe = 184
makeDiceProfile(id, all_dice_profiles, idxs, idxe, outdir)



##################################################################
# Plot the corresponding labels in the image slices from the CT test samples


def plotContourComparison(id, idx, r, xs, xe, ys, ye, plotvariants, colors, linewidth=1, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    #linewidth = 1
    
    # Plot logic
    img, header = nrrd.read(os.path.join(ct_root, "test", id + ".nrrd"))
    imslice = img[xs:-xe, ys:-ye, idx]
    
    #ax.set_title("CT test sample " + id + ", slice " + str(idx))
    ax.imshow(imslice, cmap="gray", alpha=1.)
    # write title in bottom right in white
    txt = id + ", slice " + str(idx)
    ax.text(.01, 0.01, txt, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, color='white')
    # explanation of arguments of ax.text
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html
    
    # make text on left bottom of image
    txt = "r = " + str(round(r, 2))
    ax.text(.999, 0.01, txt, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, color='white')    

    contours = {}

    # GT
    lblslice = all_labels[id]['GT']['GT'][xs:-xe, ys:-ye, idx]
    contours["GT"] = ax.contour(lblslice, levels=[0.5], colors=colors[0], linewidths=linewidth, linestyles='solid')

    for j,variant in enumerate(plotvariants):
        if "CT" in variant:
            category = "CT/CT"
        else:
            category = "LPD/LPD"
        lblslice = all_labels[id][category][variant][xs:-xe, ys:-ye, idx]
        contours[variant] = ax.contour(lblslice, levels=[0.5], colors=colors[1+j], linewidths=linewidth, linestyles='solid')
    
    ax.axis('off')
    
    # Legend
    legend_labels = [f'{label}' for label in contours]
    legend_handles = [plt.Line2D([], [], color=contour.collections[0].get_edgecolor()) for contour in contours.values()]
    ax.legend(legend_handles, legend_labels, loc='upper right')

# Show labels as contours instead of filled
# one color per label
colors = ["tab:red", "tab:cyan", "tab:purple", "turquoise", "red", "orange"]
variants = ["2.5D 5-pos 5-slices", "2D 5-pos", "2.5D 9-pos 3-slices", "2D 9-pos"]

id = "002753"
df = pd.read_csv("/home/aime/monai/" + id + "_Dice-profiles.csv")
# find connected components in the GT and label them
GT = df["GT"] > 0
GT, nregions = label(GT)
gidx = np.where(GT == 5)
idxs = gidx[0][0]
idxe = gidx[0][-1]

varlist = []
plotvariants = ["CT", "2D 9-pos"]
varlist.append(plotvariants)
plotvariants = ["2.5D 9-pos 3-slices", "2D 9-pos"]
varlist.append(plotvariants)
plotvariants = ["2.5D 5-pos 5-slices", "2D 5-pos"]
varlist.append(plotvariants)

outdir = "/home/aime/monai/visualisations/labelling"
for k, plotvariants in enumerate(varlist):
    for idx in range(idxs-3, idxe+3):
        r = (idx-idxs)/(idxe-idxs)
        plotContourComparison(id, idx, r, 35, 30, 30, 50, plotvariants, colors, linewidth=2)
        plt.savefig(os.path.join(outdir, id + "_labelling-" + str(k+1) + "_s" + str(idx) + ".png"),dpi=300,bbox_inches='tight', pad_inches=0)
        plt.close()



# # Example usage in a subplot
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# plot_dice_scores("002753", all_dice_profiles, 125, 225, ax=axs[0])
# plot_dice_scores("002554", all_dice_profiles, 125, 225, ax=axs[1])
# plt.show()

# plot_dice_scores("002554", all_dice_profiles, 148, 165)
# plt.show()
