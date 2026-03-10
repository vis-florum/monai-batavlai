from monai.metrics import DiceMetric, MeanIoU
import torch
import os
import nibabel as nib
import numpy as np
import sys
from pathlib import Path
import nrrd
#import SimpleITK as sitk


pred_dir = sys.argv[1]
#pred_dir = "/media/Store-SSD/Stembank/pine-LPDrecon/2.5D_LPD_reconstructions_5_src_pos_5_cons_slices_middle/test/labels/bata-unet-4steps_LPDsample-2.5D-5src-5pos-mid/"
GT_dir = os.path.join(Path(pred_dir).parent, "final")

# only list .nii or .nii.gz files
#pred_list = sorted([f for f in os.listdir(pred_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
#GT_list = sorted([f for f in os.listdir(GT_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
pred_list = sorted([f for f in os.listdir(pred_dir) if f.endswith('.nrrd')])
GT_list = sorted([f for f in os.listdir(GT_dir) if f.endswith('.nrrd')])


dices = []
ious = []
for n in range(len(pred_list)):
    y_pred, _ = nrrd.read(os.path.join(pred_dir, pred_list[n]))
    y, _ = nrrd.read(os.path.join(GT_dir, GT_list[n]))
    y_pred = np.array(y_pred)
    y = np.array(y)
    
    # For nifti:
    #y_pred = nib.load(os.path.join(pred_dir, pred_list[n]))
    #y = nib.load(os.path.join(GT_dir, GT_list[n]))

    #y_pred = np.array(y_pred.dataobj)
    #y = np.array(y.dataobj)

    # convert to binary masks
    y_pred[y_pred > 0] = 1
    y[y > 0] = 1

    y_pred = torch.from_numpy(y_pred)
    y = torch.from_numpy(y)


    # Initialize the Metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    iou_metric = MeanIoU(include_background=False, reduction="mean")

    # Add batch and channel dimension not present
    if len(y_pred.shape) == 3:
        y_pred = y_pred.unsqueeze(0)
        y_pred = y_pred.unsqueeze(0)
        y = y.unsqueeze(0)
        y = y.unsqueeze(0)

    # Convert to one-hot format if necessary
    # ...

    # Compute metrics
    dice_score = dice_metric(y_pred=y_pred, y=y)
    iou_score  = iou_metric(y_pred=y_pred, y=y)
    dices.append(dice_score.item()) # make skalar
    ious.append(iou_score.item())

print(dices)
print(np.mean(dices))
print(ious)
print(np.mean(ious))

# write to txt file
outfile = os.path.join(pred_dir, "dice_scores.txt")
with open(outfile, "w") as output:
    output.write("Mean dices: \n")
    output.write(str(dices))
    output.write("\n")
    output.write("Mean: \n")
    output.write(str(np.mean(dices)))

outfile = os.path.join(pred_dir, "iou_scores.txt")
with open(outfile, "w") as output:
    output.write("Mean ious: \n")
    output.write(str(ious))
    output.write("\n")
    output.write("Mean: \n")
    output.write(str(np.mean(ious)))