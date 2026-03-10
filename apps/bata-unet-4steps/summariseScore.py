# summarise the scores saved in the txt files

import os
import numpy as np
import sys
from pathlib import Path

variant = 1 # <----------------- CHANGE HERE

if variant == 1:
    scorefilename = "dice_scores.txt"
elif variant == 2:
    scorefilename = "iou_scores.txt"

lpd_rootdir = "/media/Store-SSD/Stembank/pine-LPDrecon/nrrd/"
lpd_variants = [d for d in os.listdir(lpd_rootdir) if os.path.isdir(os.path.join(lpd_rootdir, d))]

# nested dicts containing info about train/val scores of each unet
# LPD UNETs on themselves:
scores_unetLPD = {lpd_variant: {"val": [], "test": []} for lpd_variant in lpd_variants}
# CT UNET on LPD reconstructions:
scores_unetCT =  {lpd_variant: {"val": [], "test": []} for lpd_variant in lpd_variants}
# CT UNET on full CT reconstructions:
scores_unetFullCT =  {"val": [], "test": []}


outfile = "/home/aime/monai/" + scorefilename

##########################################################################################
# Each LPD variant
for lpd_variant in lpd_variants:
    #print(lpd_variant)
    lpd_dir = os.path.join(lpd_rootdir, lpd_variant)
    
    # Each set
    for set in ["val", "test"]:
        workdir = os.path.join(lpd_dir, set, "labels")
        
        modeldirs = [d for d in os.listdir(workdir) if os.path.isdir(os.path.join(workdir, d))]
        # only keep if modeldir contains "UNET"
        modeldirs = [d for d in modeldirs if "UNET" in d]
        
        for modeldir in modeldirs:
            modeldir_full = os.path.join(workdir, modeldir)
            scorefile = os.path.join(modeldir_full, scorefilename)    
            with open(scorefile, "r") as f:
                lines = f.readlines()
                scores = lines[1].strip()  # remove \n
                # Convert the second line string to a list of floats
                scores = scores.strip('[]').split(',')
                scores = [float(element) for element in scores]
           
                if "CT" in modeldir:
                    scores_unetCT[lpd_variant][set].append(scores)
                else:
                    scores_unetLPD[lpd_variant][set].append(scores)


#################################################################################
# Full CT reconstruction
CTsample_dir = "/media/Store-SSD/Stembank/pine/LPDsample"

for set in ["val", "test"]:
    workdir = os.path.join(CTsample_dir, set, "labels","UNET_LPDsample-CT")
    
    scorefile = os.path.join(workdir, scorefilename)
    
    with open(scorefile, "r") as f:
        lines = f.readlines()
        scores = lines[1].strip()  # remove \n
        scores = scores.strip('[]').split(',')
        scores = [float(element) for element in scores]
        
        scores_unetFullCT[set].append(scores)


#################################################################################
def formatWriteOut(outfile,scores,lpdvariant=None):
    with open(outfile, "a") as output:
        if lpdvariant:
            output.write(lpdvariant + ":\n")
        output.write("val: ")
        output.write(str(np.round(scores["val"][0], 3)) + "\n")
        output.write("test: ")
        output.write(str(np.round(scores["test"][0], 3)) + "\n")
        output.write("Mean val: ")
        output.write(str(np.round(np.mean(scores["val"][0]), 3)) + "\n")
        output.write("Mean test: ")
        output.write(str(np.round(np.mean(scores["test"][0]), 3)) + "\n")
        output.write("\n")
        

# Write arrays and means
with open(outfile, "w") as output:
    output.write("################################################## \n")
    output.write("CT-trained UNET infers on full CT reconstructions: \n")
    output.write("################################################## \n")
formatWriteOut(outfile, scores_unetFullCT)

with open(outfile, "a") as output:
    output.write("################################################## \n")
    output.write("LPD-trained UNETs infer on LPD reconstructions: \n")
    output.write("################################################## \n")
for lpd_variant in lpd_variants:
    formatWriteOut(outfile, scores_unetLPD[lpd_variant],lpd_variant)
    
with open(outfile, "a") as output:
    output.write("################################################## \n")
    output.write("CT-trained UNET infers on LPD reconstructions: \n")
    output.write("################################################## \n")
for lpd_variant in lpd_variants:
    formatWriteOut(outfile, scores_unetCT[lpd_variant],lpd_variant)


import pandas as pd
import matplotlib.pyplot as plt

# all in one dict:
scores_dict = {
    'CT train, CT infer': {"CT": scores_unetFullCT},
    'LPD train, LPD infer': scores_unetLPD,
    'CT train, LPD infer': scores_unetCT,
}


# Creating the DataFrame
scores_df = pd.DataFrame.from_dict({(i,j): scores_dict[i][j] 
                           for i in scores_dict.keys() 
                           for j in scores_dict[i].keys()},
                       orient='index')
# replace val and test values by their means
scores_df["val"] = scores_df["val"].apply(lambda x: np.mean(x))
scores_df["test"] = scores_df["test"].apply(lambda x: np.mean(x))
scores_df = scores_df.round(3)

# Replace the LPD variant names by short new names
# detect whether there is 2.5 or 2 in LPD variant name and the number of src_pos and cons_slices
import re
pt_src = re.compile(r'\d+_src_pos')
pt_slices = re.compile(r'\d+_cons_slices')
lpd_variants_short = []
for lpd_variant in lpd_variants:
    src_pos = pt_src.search(lpd_variant).group()
    if "2.5D" in lpd_variant:
        cons_slices = pt_slices.search(lpd_variant).group()
        lpd_variants_short.append("2.5D " + src_pos[0] + "-pos " + cons_slices[0] + "-slices")
    elif "2D" in lpd_variant:
        lpd_variants_short.append("2D " + src_pos[0] + "-pos ")

# Replace the LPD variant names by short new names
scores_df = scores_df.rename(index={lpd_variants[i]: lpd_variants_short[i] for i in range(len(lpd_variants))})

scores_df = scores_df.rename(index={"CT train, CT infer": "CT-CT",
                                    "CT train, LPD infer": "CT-LPD",
                                    "LPD train, LPD infer": "LPD-LPD"})

# Apply a color to each 'Trained On' category
colors = {
    'CT train, CT infer': 'lightgreen',
    'CT train, LPD infer': 'lightblue',
    'LPD train, LPD infer': 'lightcoral'
}

# Latex table
latex_df = scores_df.transpose()
latex_table = latex_df.to_latex(index=True,
                                caption="Your Table Caption",
                                label="tab:your_label",
                                float_format="%.3f",
                                bold_rows=True,
                                multicolumn=True,
                                )

with open('table.tex', 'w') as file:
    file.write(latex_table)