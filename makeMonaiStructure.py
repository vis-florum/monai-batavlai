
import os
#import sys
import glob
import numpy as np
import nibabel as nib
import nrrd
import matplotlib.pyplot as plt

files_train  = ["002943.npy", "002263.npy", "001962.npy", "002754.npy", "002741.npy", "002341.npy", "002951.npy",
                "002312.npy", "000961.npy", "003261.npy", "000952.npy", "001064.npy", "002051.npy", "001852.npy",
                "003012.npy", "003241.npy", "002642.npy", "002143.npy", "001224.npy", "002031.npy", "001761.npy",
                "003014.npy", "002663.npy", "002061.npy", "001342.npy", "002622.npy", "002313.npy", "001022.npy",
                "002733.npy", "001053.npy", "001121.npy", "002211.npy", "003143.npy", "001611.npy", "001821.npy",
                "003253.npy", "003362.npy", "001152.npy", "001322.npy", "002523.npy", "002543.npy", "003053.npy"]
files_val = ["002431.npy","000833.npy","000922.npy","000462.npy","000532.npy"]
files_test = ["001054.npy", "002554.npy", "002753.npy","000552.npy"]

lpdserver = "/media/Store-SSD/Stembank/pine-LPDrecon"
lpdserver_raw = os.path.join(lpdserver, "raw")
    
stembank = "/media/Store-SSD/Stembank/pine/server"
stembank_list = glob.glob(os.path.join(stembank, '*.nrrd'))


def moveSample(file_id, sampledir, stembank, setname):
    # image_src = os.path.join(stembank, file_id + ".nii")
    # label_src = os.path.join(stembank, "labels", "final", file_id + ".nii.gz")
    image_src = os.path.join(stembank, file_id + ".nrrd")
    label_src = os.path.join(stembank, "labels", "final", file_id + ".nrrd")
    
    # image_target = os.path.join(sampledir, setname, file_id + ".nii")
    # label_target = os.path.join(sampledir, setname, "labels", "final", file_id + ".nii.gz")
    image_target = os.path.join(sampledir, setname, file_id + ".nrrd")
    label_target = os.path.join(sampledir, setname, "labels", "final", file_id + ".nrrd")
    # os.system("cp " + image_src + " " + image_target)
    # os.system("cp " + label_src + " " + label_target)
    # create symbolic links instead of copying
    os.system("ln -s " + image_src + " " + image_target)
    os.system("ln -s " + label_src + " " + label_target)

# determine whether file belongs to train or test set or validation set
def getSetName(file_id,files_train,files_val,files_test):
    ids_train = sorted([f.split(".")[0] for f in files_train])  
    ids_val   = sorted([f.split(".")[0] for f in files_val])
    ids_test  = sorted([f.split(".")[0] for f in files_test])
    
    if file_id in ids_train:
        return "train"
    elif file_id in ids_val:
        return "val"
    elif file_id in ids_test:
        return "test"
    else:
        return "unknown"

def convert_to_8bit(data):
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data


def transformAndSplit(lpd_dir, files_train, files_val, files_test):
    '''
    Transforms the LPD images to correct NNRD format 
    and splits them into train, val and test sets
    '''
    npy_files = sorted([f for f in os.listdir(lpd_dir) if os.path.isfile(os.path.join(lpd_dir, f))])

    for npy_file in npy_files:
        file_id = npy_file.split("_")[0]
        print(file_id)
        
        # Determine target directory
        setname = getSetName(file_id, files_train, files_val, files_test)
        lpd_variant = lpd_dir.split("/")[-1]
        target_dir = os.path.join(lpdserver, "nrrd", lpd_variant, setname)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        # Load reconstructed LPD image
        lpd_img = np.load(os.path.join(lpd_dir, npy_file))
        lpd_img = np.transpose(lpd_img, (1,0,2))    # transpose 2D slices due to some error in the LPD reconstruction
        lpd_img = convert_to_8bit(lpd_img)
        
        # Load corresponding stembank image
        # nii_file = os.path.join(stembank, file_id + ".nii")
        # nii_image = nib.load(nii_file)
        nrrd_file = os.path.join(stembank, file_id + ".nrrd")
        nrrd_image, header = nrrd.read(nrrd_file)

        # Transform to NIfTI with correct affine and save in target directory
        # if nii_image.header.get_data_shape() == lpd_img.shape:
        #     new_nii_image = nib.Nifti1Image(lpd_img, nii_image.affine)
        #     new_nii_image.header.set_xyzt_units('mm', 'sec')
        #     outpath = os.path.join(target_dir, file_id + ".nii")
        #     nib.save(new_nii_image, outpath)
        if nrrd_image.shape == lpd_img.shape:
            lpd_nrrd_image = os.path.join(target_dir, file_id + ".nrrd")
            nrrd.write(os.path.join(target_dir, file_id + ".nrrd"), lpd_img, header)
        else:
            print("Size mismatch: " + npy_file)


def splitLabels(lpd_dir, files_train, files_val, files_test):
    '''
    Splits the labels into train, val and test sets
    '''
    npy_files = sorted([f for f in os.listdir(lpd_dir) if os.path.isfile(os.path.join(lpd_dir, f))])

    for npy_file in npy_files:
        file_id = npy_file.split("_")[0]
        print(file_id)
        
        # Determine target directory
        setname = getSetName(file_id, files_train, files_val, files_test)
        lpd_variant = lpd_dir.split("/")[-1]
        target_dir = os.path.join(lpdserver, "nrrd", lpd_variant, setname, "labels/final")
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        # Populate label directory
        # label_src = os.path.join(stembank, "labels", "final", file_id + ".nii.gz")
        label_src = os.path.join(stembank, "labels", "final", file_id + ".nrrd")
        
        # label_target = os.path.join(target_dir, file_id + ".nii.gz")
        label_target = os.path.join(target_dir, file_id + ".nrrd")
        
        # os.system("cp " + label_src + " " + label_target)      
        os.system("ln -s " + label_src + " " + label_target)    # create symbolic links instead of copying


########################################

def makeFullCT():
    # Pull from stembank full CT and split into train, val and test sets
    sampledir = "/media/Store-SSD/Stembank/pine/LPDsample-nrrd"
    if not os.path.exists(sampledir):
        os.makedirs(sampledir)
    # make subdirectories
    setdirs = ["train", "val", "test"]
    for setdir in setdirs:
        if not os.path.exists(os.path.join(sampledir, setdir, "labels", "final")):
            os.makedirs(os.path.join(sampledir, setdir, "labels", "final"))
        

    train_ids = [f.split(".")[0] for f in files_train]
    val_ids = [f.split(".")[0] for f in files_val]
    test_ids = [f.split(".")[0] for f in files_test]

    for id in train_ids:
        moveSample(id, sampledir, stembank, "train")

    for id in val_ids:
        moveSample(id, sampledir, stembank, "val")
        
    for id in test_ids:
        moveSample(id, sampledir, stembank, "test")
    

def makeLPD(lpdvar="all"):
    # Pull from LPD reconstruction directory and split into train, val and test sets
    lpd_variants = [d for d in os.listdir(lpdserver_raw) if os.path.isdir(os.path.join(lpdserver_raw, d))]

    for lpd_variant in lpd_variants:
        lpd_dir = os.path.join(lpdserver_raw, lpd_variant)
        if lpd_variant == lpdvar or lpdvar == "all":
            print("Treating LPD variant: " + lpd_variant)
            transformAndSplit(lpd_dir, files_train, files_val, files_test)
            splitLabels(lpd_dir, files_train, files_val, files_test)


############################################################
def main():
    # makeFullCT()
    makeLPD("2D_UNET_reconstructions_5_src_pos")


if __name__ == "__main__":
    main()