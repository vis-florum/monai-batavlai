#%%
from torch.utils.tensorboard import SummaryWriter
 
#%% Import App and set Study

import sys
import os
import matplotlib.pyplot as plt
import importlib    # For updating modules
import numpy as np

app_dir = "/home/aime/monai/apps/bata-unet-4steps_LPDsample-CT"

sys.path.append(app_dir)
import main
importlib.reload(main)
from main import MyApp

from pathlib import Path
from monailabel.utils.others.generic import device_list, file_ext

studies = "/media/Store-SSD/Stembank/pine/server-testing"
studies = "/media/Store-SSD/Stembank/pine/LPDsample/train"
studies = "/media/Store-SSD/Stembank/pine/LPDsample-nrrd/train"

conf = {
        "models": "segmentation",
        "preload": "true",
    }

app = MyApp(app_dir, studies, conf)

# Load Dada:
D = app._datastore.datalist()
d0 = D[0]
d1 = D[17]  # different spacing
d1

val_studies = "/media/Store-SSD/Stembank/pine/LPDsample-nrrd/val"
from monailabel.datastore.local import LocalDatastore 
ds_val = LocalDatastore(val_studies,extensions=["*.nrrd"])


#%% Plain load a .nii and check its space
import nibabel as nib
import nrrd
import numpy as np

nii_path = d0["image"]
nii_image = nib.load(nii_path)
data = nii_image.get_fdata()
data = data.astype(np.uint16)
nii_header = nii_image.header
spacing = np.sqrt(np.sum(nii_image.affine[:3, :3]**2, axis=0))
translation = nii_image.affine[:3, 3]
translation
nii_header.get_zooms()
nrrd_header = {}
nrrd_header['spacings'] = spacing.tolist()
nrrd_header['units'] = ['mm', 'mm', 'mm']


nrrd_path = '/media/Store-SSD/monaitest/test1.nrrd'
nrrd.write(nrrd_path, data, header=nrrd_header)


#%% Inference
import pandas as pd


# Full infer to file:
# image_id = d0["image"]
# device = device_list()[0]
# res = app.infer(request={"model": "segmentation", "image": image_id, "device": device}) # writes to tmp


# Inference live
# myInf = app.init_infers().get(model)
# myInf = app.init_infers()['segmentation']
model = "segmentation"
myInf = app._infers.get(model)

d0_pre = myInf.run_pre_transforms(transforms=myInf.pre_transforms(),data=d0)
d0_pre["image"].shape
plt.imshow(d0_pre["image"][0, :, :, 10], cmap="gray")
plt.savefig('pre.png')

pd.DataFrame(d0_pre["image_meta_dict"].get("affine"))
d0_pre["image_meta_dict"].get("space")
# Centres the volume and inverts the X-axis (RAS??) !!!


# print all key and value pairs as newlines
for key, value in d0_pre["image_meta_dict"].items():
    print(key, ' : ', value)

d0_inf = myInf.run_inferer(data=d0_pre)
d0_inf["pred"].shape    # one channel for each label
img = d0_inf["pred"][0, :, :, 10].cpu()
np.unique(img)
plt.imshow(img, cmap="gray")
plt.savefig('inf.png')




# importlib.reload(lib.transforms.transforms)
# from lib.transforms.transforms import *
# d0_inf_uncrop = Uncropd(keys=("image"),ref_image="image")(d0_inf)
# d0_inf_uncrop["image"].shape
# d0_inf_uncrop["pred"].shape

d0_post = myInf.run_post_transforms(transforms=myInf.post_transforms(),data=d0_inf)
d0_post["pred"].shape

pd.DataFrame(d0_post["image_meta_dict"].get("affine"))
pd.DataFrame(d0_post["pred_meta_dict"].get("affine"))
d0_post["image_meta_dict"].get("space")
d0_post["pred_meta_dict"].get("space")

for key, value in d0_post["pred_meta_dict"].items():
    print(key, ' : ', value)
    
img = d0_post["pred"][:, :, 10].cpu()
np.unique(img)
plt.imshow(img, cmap="gray")
plt.savefig('post.png')

myInf.writer(d0_post,extension=".nrrd") # Writes to tmp???

#%% Training
model = "segmentation"
myTask = app._trainers.get(model)

from monailabel.tasks.train.basic_train import Context
context: Context = Context()
context.__init__()  # load standard values



request={
            "model": "segmentation",
            "max_epochs": 2,     # JH changed <---- not followed by slicer
            "dataset": "Dataset",  # PersistentDataset, CacheDataset
            "train_batch_size": 1,  # Logs can be differently long??
            "val_batch_size": 1,
            "multi_gpu": False,  # JH changed
            "val_split": 0.1,
            # "train_ds": "/media/Store-SSD/Stembank/pine/LPDsample/train/datastore_v2.json",  # same as where server resides?
            # "val_ds": "/media/Store-SSD/Stembank/pine/LPDsample/val/datastore_v2.json",
        }
import time

request["device"] = "cuda"  # if no multi-gpu

from datetime import datetime
request["run_id"] = datetime.now().strftime("%Y%m%d_%H%M%S")

context.run_id = request["run_id"]
context.request = request
context.datalist = app._datastore.datalist()
context.multi_gpu = request["multi_gpu"]
context.network, context.optimizer = myTask._create_network_and_optimizer(context)

from monailabel.tasks.train.basic_train import *
context.device = myTask._device(context)
context.max_epochs = request["max_epochs"]
context.train_batch_size = request["train_batch_size"]
context.val_batch_size = request["val_batch_size"]
context.pretrained = request["pretrained"]


datastore = app._datastore

datalist = myTask.pre_process(request, datastore)


context.dataset_type = request["dataset"]
context.dataloader_type = request["dataloader"]



name = "debugging"
context.output_dir = os.path.join(myTask._model_dir, name)
context.cache_dir = os.path.join(context.output_dir, f"cache_{context.run_id}")
context.events_dir = os.path.join(context.output_dir, f"events_{context.run_id}")

tracking_uri = request.get("tracking_uri", myTask._tracking_uri)
if not tracking_uri:
    tracking_uri = path_to_uri(os.path.join(context.output_dir, "mlruns"))
experiment_name = request.get("tracking_experiment_name")
experiment_name = experiment_name if experiment_name else request.get("model")
run_name = request.get("tracking_run_name")
run_name = run_name if run_name else f"run_{context.run_id}"

context.tracking = request.get("tracking", myTask._tracking)
context.tracking = context.tracking[0] if isinstance(context.tracking, list) else context.tracking
context.tracking_uri = tracking_uri
context.tracking_experiment_name = experiment_name
context.tracking_run_name = run_name


start_ts = time.time()
context.start_ts = start_ts

# investigate datalist
len(context.datalist)
context.train_datalist, context.val_datalist = myTask.partition_datalist(context)
len(context.train_datalist)
len(context.val_datalist)


#tdl = myTask.train_data_loader(context=context)
#vdl = myTask.val_data_loader(context=context)


context.network, context.optimizer = myTask._create_network_and_optimizer(context)
context.evaluator = myTask._create_evaluator(context)
context.trainer = myTask._create_trainer(context)

myTask.finalize(context)

meta_tracking = True
set_track_meta(True)

context.trainer.run()       # here it fails
from ignite.engine.engine import Engine
Engine.run(context.trainer,data=context.trainer.data_loader, max_epochs=context.trainer.state.max_epochs, epoch_length=context.trainer.state.epoch_length)

len(context.trainer.data_loader.dataset.data)

#ll = 18
debug_data_loader = context.trainer.data_loader
debug_data_loader.dataset.data = debug_data_loader.dataset.data[16:]
len(debug_data_loader.dataset.data)
context.trainer.state.epoch_length = len(debug_data_loader.dataset.data)
Engine.run(context.trainer,data=debug_data_loader, max_epochs=1)

context.trainer.state


context.trainer.state.dataloader = context.trainer.data_loader
context.trainer._internal_run()

# Check its state
context.trainer.state
context.trainer.state.iteration     # stuck at iteration 7
A = context.trainer.state.batch[0]
A
A.keys()
A['image'].shape
A['label'].shape

B = context.trainer.state.batch["image"].cpu()
from matplotlib import pyplot as plt
plt.imshow(B[0,0,:,:,10])
plt.savefig('engine-iter7-2.png')   # file 753! the last file in the list of train data

debug_data_loader.dataset.dataset.data[-1]

context.evaluator.state
context.evaluator


import torch
from torch.cuda import empty_cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

from monailabel.tasks.train.handler import PublishStatsAndModel, prepare_stats
prepare_stats(start_ts, context.trainer, context.evaluator)

myTask.cleanup(request)

#dataset, datalist = myTask._dataset(context, context.train_datalist, is_train=True)
#dataset, datalist = myTask._dataset(context, context.val_datalist, is_train=False)



#%% Transforms
preTrafo = myTask._validate_transforms(myTask.train_pre_transforms(context=context), "Training", "pre")
d0_pre = preTrafo(d0)
d1_pre = preTrafo(d1)
d0_pre['image'].shape
d1_pre['image'].shape
plt.imshow(d0_pre['image'][0,:,:,10].cpu())
plt.savefig('train-pre.png')
plt.imshow(d1_pre['image'][0,:,:,492].cpu())
plt.savefig('train-pre-753.png')
plt.imshow(d0_pre['image'][0,:,100,:].cpu())
plt.savefig('train-pre-side.png')
plt.imshow(d1_pre['image'][0,:,100,:].cpu())
plt.savefig('train-pre-753-side.png')

preTrafoVal = myTask._validate_transforms(myTask.val_pre_transforms(context=context), "Validation", "pre")
d0_preVal = preTrafoVal(d0)
d1_preVal = preTrafoVal(d1)
d0_preVal['image'].shape
d1_preVal['image'].shape
plt.imshow(d0_preVal['image'][0,:,:,10].cpu())
plt.savefig('val-pre.png')
plt.imshow(d1_preVal['image'][0,:,:,10].cpu())
plt.savefig('val-pre-753.png')

inferer = myTask.train_inferer(context=context)
input = d0_pre["image"]
input = input.unsqueeze(0) # add dimension in front to A
input = input.to("cuda")   # to GPU
d0_inf = inferer(inputs=input, network=context.network).data # not a dict!
input = d1_pre["image"].unsqueeze(0).to("cuda")
d1_inf = inferer(inputs=input, network=context.network).data # not a dict!
d0_inf.shape
img = d0_inf[0,0,:,:,10].cpu()
np.unique(img)
img.max()
img.min()
plt.imshow(img)
plt.savefig('train-inf.png')

d0["pred"] = d0_inf[0,:,:,:,:]
d1["pred"] = d1_inf[0,:,:,:,:]


inferer = myTask.val_inferer(context=context)
inputVal = d0_preVal["image"]
inputVal = inputVal.unsqueeze(0) # add dimension in front to A
inputVal = inputVal.to("cuda")   # to GPU
d0_infVal = inferer(inputs=inputVal, network=context.network).data # not a dict!
d0_infVal.shape
img = d0_infVal[0,0,:,:,10].cpu()
np.unique(img)
img.max()
img.min()
plt.imshow(img)
plt.savefig('val-inf.png')


d0_inf_dict = d0_pre.copy()
d0_inf_dict["pred"] = d0_inf[0,:,:,:,:]

d1_inf_dict = d1_pre.copy()
d1_inf_dict["pred"] = d1_inf[0,:,:,:,:]

d0_infVal_dict = d0_preVal.copy()
d0_infVal_dict["pred"] = d0_infVal[0,:,:,:,:]

postTrafo = myTask._validate_transforms(myTask.train_post_transforms(context=context), "Training", "post")
d0_post = postTrafo(d0_inf_dict)
d0_post['image'].shape
d0_post['pred'].shape
d0_post['label'].shape
img = d0_post["pred"][0, :, :, 10].cpu()
np.unique(img)
plt.imshow(img)
plt.savefig('train-post.png')
plt.imshow(d0_post["label"][0, :, :, 10].cpu())
plt.savefig('train-post-label.png')
plt.imshow(d0_post["label"][1, :, :, 10].cpu())
plt.savefig('train-post-label.png')

d1_post = postTrafo(d1_inf_dict)
d1_post['image'].shape
d1_post['pred'].shape
d1_post['label'].shape
img = d1_post["pred"][0, :, :, 10].cpu()
np.unique(img)
plt.imshow(img)
plt.savefig('train-post-753.png')
plt.imshow(d1_post["label"][1, :, :, 492].cpu())
plt.imshow(d1_post["image"][0, :, :, 492].cpu())
plt.savefig('train-post-label-753.png')


postTrafoVal = myTask._validate_transforms(myTask.val_post_transforms(context=context), "Validation", "post")
d0_postVal = postTrafoVal(d0_infVal_dict)
d0_postVal['image'].shape
d0_postVal['pred'].shape
d0_postVal['label'].shape
img = d0_postVal["pred"][0, :, :, 10].cpu()
np.unique(img)
plt.imshow(img)
plt.savefig('val-post.png')
plt.imshow(d0_postVal["label"][0, :, :, 10].cpu())
plt.savefig('val-post-label.png')
plt.imshow(d0_postVal["label"][1, :, :, 10].cpu())
plt.savefig('val-post-label.png')


# scoring
han = myTask.train_key_metric(context=context)['train_mean_dice']
han.compute(context)
han = myTask.train_handlers(context=context)


os.putenv("MASTER_ADDR", "127.0.0.1")
os.putenv("MASTER_PORT", "1234")

app.train(request=request)

app.train(    # AttributeError: 'MyApp' object has no attribute '_trainers'???????
        request={
            "model": "segmentation",
            "max_epochs": 2,     # JH changed <---- not followed by slicer
            "dataset": "Dataset",  # PersistentDataset, CacheDataset
            "train_batch_size": 1,  # Logs can be differently long??
            "val_batch_size": 1,
            "multi_gpu": False,  # JH changed
            "val_split": 0.1,
        },
    )





#%%
import json
import logging
import os
from typing import Dict

# change current working dir to the app dir (for debugging)
#os.chdir("apps/bata-unet")
os.getcwd()

from lib.infers.segmentation import Segmentation
from lib.trainers.segmentation import Segmentation
from lib.transforms.transforms import GetOriginalInformation


from monai.networks.nets import UNet



from  monai.transforms import *
import matplotlib.pyplot as plt
import numpy as np
from fast_histogram import histogram1d

from lib.transforms.transforms import NormalizeLabelsInDatasetd

# %% Run as in main

import json
import logging
import os
from typing import Dict

# change current working dir to the app dir (for debugging)
#os.chdir("apps/bata-unet")
os.getcwd()

import lib.configs  # need to be in folder for that to work
from lib.activelearning import Last
#from lib.infers.deepgrow_pipeline import InferDeepgrowPipeline     # JH: sth doesn't work when loading this
#from lib.infers.vertebra_pipeline import InferVertebraPipeline      # JH: sth doesn't work when loading this

import monailabel
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
#from monailabel.scribbles.infer import GMMBasedGraphCut, HistogramBasedGraphCut
from monailabel.tasks.activelearning.first import First
from monailabel.tasks.activelearning.random import Random

from monai.data import (
    CacheDataset,
    DataLoader,
    Dataset,
    PersistentDataset,
    SmartCacheDataset,
    ThreadDataLoader,
    get_track_meta,
    partition_dataset,
    set_track_meta,
)

from monailabel.tasks.train.basic_train import BasicTrainTask, Context
import torch

from monailabel.tasks.train.basic_train import BasicTrainTask, Context
import torch
from monai.transforms import Compose
import os
import matplotlib.pyplot as plt
from  monai.transforms import *
import importlib

import lib.trainers.segmentation
importlib.reload(lib.trainers.segmentation)
from lib.trainers.segmentation import Segmentation as TrainSegmentation

import lib.infers.segmentation
importlib.reload(lib.infers.segmentation)
from lib.infers.segmentation import Segmentation as InferSegmentation

import lib.configs.segmentation
importlib.reload(lib.configs.segmentation)
from lib.configs.segmentation import Segmentation as ConfigSegmentation

#torch.cuda.get_device_name(0)

imgdir = "/home/aime/monai/pine/server-testing/"
labeldir = "/home/aime/monai/pine/server-testing/labels/final/"
nii_files = [f for f in os.listdir(imgdir) if f.endswith('.nii') and os.path.isfile(os.path.join(imgdir, f))]
nii_files.sort()
# Dict of images and labels
D = [{"image": os.path.join(imgdir, f), "label": os.path.join(labeldir, f.replace(".nii",".nii.gz"))} for f in nii_files]
d0 = D[0]


d = LoadImaged(keys=("image", "label"))(d0)
pd.DataFrame(d["image_meta_dict"].get("affine"))
d["image_meta_dict"]
d["image_meta_dict"].get("space")
d["label_meta_dict"].get("space")
d["image_meta_dict"].get("srow_x")
d["image_meta_dict"].keys()
# save as nrrd
import nrrd
nrrd.write('test.nrrd', d["image"].cpu().numpy(), d["image_meta_dict"])

d

d = LoadImaged(keys=("image", "label"))(d1)
targetSpacing = myTask.target_spacing
targetSpacing = (1.3671875, 1.3671875, 10.0)
targetSpacing = (1.3671875, 1.3671875)
targetSpacing = (1.5625, 1.5625, 10.0)
d = Spacingd(keys=("image", "label"), pixdim=targetSpacing, mode=("bilinear", "nearest"))(d)
d["image_meta_dict"]["spatial_shape"]
d["image"].shape
d["image_meta_dict"].keys()
d["image_meta_dict"]["dim"]
d["image_meta_dict"]["pixdim"]
d["image_meta_dict"]["affine"]
d["image_meta_dict"]["original_affine"]

from lib.transforms.transforms import Unpadd
from monailabel.transform.post import Restored

roi_size = (256, 256, 512)
labels = {"liveknot": 3,}
d = EnsureChannelFirstd(keys=("image", "label"))(d)
#d = NormalizeLabelsInDatasetd(keys="label", label_names=labels),  # Specially for missing labels
d = EnsureTyped(keys=("image", "label"))(d)
d = Spacingd(keys=("image", "label"), pixdim=targetSpacing, mode=("bilinear", "nearest"))(d)

dc = CenterSpatialCropd(keys=("image", "label"), roi_size=(roi_size[0],roi_size[1],-1))(d)  # JH: crop symmetrically from middle, ignore length direction

dc = ResizeWithPadOrCropd(keys=("image", "label"), spatial_size=roi_size, mode="constant", method="end", constant_values=0)(d)


d["image"].shape
dc["image"].shape

plt.imshow(dc['image'][0,:,100,:].cpu())
plt.savefig('centercrop-1.png')

SpatialPadd(keys=("image", "label"), spatial_size=roi_size, method="end", mode="constant", constant_values=0)(d)
# ?????

plt.imshow(d['image'][:,:,10].cpu())
plt.savefig('00.png')
plt.imshow(d['label'][:,:,10].cpu())
plt.savefig('00-label.png')

S = TrainSegmentation
S._labels = {"liveknot": 3,}
S.target_spacing = (1.3671875, 1.3671875, 10.0)
S.roi_size = (256, 256, 512)

context = Context()
context.request["device"]
context.device = "cuda"

T = TrainSegmentation.train_pre_transforms(self=S, context=context)
T = Compose(T)

TD = T(D)

for i in range(len(TD)):
    print(f"{TD[i]['image'].shape} => {TD[i]['label'].shape}")

plt.imshow(TD[0]['image'][0,:,:,10].cpu())
plt.savefig('00-T.png')
L = TD[0]['label'][0,:,:,10].cpu()
plt.imshow(L)
plt.savefig('00-T-label.png')
# transpose image
plt.imshow(TD[0]['image'][0,:,10,:].cpu())


network = UNet(
            spatial_dims=3,
            in_channels=1, #self.number_intensity_ch,
            out_channels=2,  # All labels plus background
            #channels=[16, 32, 64, 128, 256],
            channels=[16, 32, 64, 128],
            #strides=[2, 2, 2, 2],
            strides=[2, 2, 2],
            num_res_units=2,
            norm="batch",
        )

NN0 = network(TD[0]['image'].unsqueeze(0).cpu())
NN0.shape

PT = TrainSegmentation.train_post_transforms(self=S, context=context)
PT = Compose(PT)

PT(NN0)

target_size = (256, 256, 512)
target_spacing = (1.3671875, 1.3671875, 10.0)


T = Segmentation.train_pre_transforms
T = SpatialPadd(keys=['image','label'], spatial_size=target_size, method="end", mode="constant", constant_values=0)
T = LoadImaged(keys=['image','label'])
T = [LoadImaged(keys=("image", "label")),
    EnsureChannelFirstd(keys=("image", "label")),
    NormalizeLabelsInDatasetd(keys="label", label_names=labels),  # Specially for missing labels
    #EnsureTyped(keys=("image", "label"), device=context.device),
    Orientationd(keys=("image", "label"), axcodes="RAS"),
    Spacingd(keys=("image", "label"), pixdim=target_spacing, mode=("bilinear", "nearest")),
    #NormalizeIntensityd(keys="image", nonzero=True),    # JH: mean centering and unit variance scaling
    # CropForegroundd(
    #     keys=("image", "label"),
    #     source_key="image",
    #     margin=10,
    #     k_divisible=[self.roi_size[0], self.roi_size[1], self.roi_size[2]],
    # ),
    #GaussianSmoothd(keys="image", sigma=0.4), # JH: Maybe deactivate
    ScaleIntensityd(keys="image"), #minv=-1.0, maxv=1.0),
    # RandSpatialCropd(
    #     keys=["image", "label"],
    #     roi_size=[self.roi_size[0], self.roi_size[1], self.roi_size[2]],
    #     random_size=False,
    # ),
    # RandRotated(
    #     keys=["image", "label"],
    #     range_x=3.141592653589793/2,  # automatically samples from (-range_x, range_x)
    #     prob=0.2,
    # ),
    # RandFlipd(keys=("image", "label"), spatial_axis=[0], prob=0.15),
    # RandFlipd(keys=("image", "label"), spatial_axis=[1], prob=0.15),
    SelectItemsd(keys=("image", "label")),  #  Select only specified items from data dictionary to release memory.
]

for f in nii_files:
    print(f)
    d = [{"image": os.path.join(imgdir, f), "label": os.path.join(labeldir, f.replace(".nii",".nii.gz"))}]
    #D = T(d[0])
    d = d[0]
    for t in T:
        d = t(d)
    print(f"{d['image'].shape} => {d['label'].shape}")
    print(d['image'].shape == d['label'].shape)
    


NN = UNet(
    spatial_dims=3,
    in_channels=1, #self.number_intensity_ch,
    out_channels=1 + 1,  # All labels plus background
    #channels=[16, 32, 64, 128, 256],
    channels=[16, 32, 64, 128],
    #strides=[2, 2, 2, 2],
    strides=[2, 2, 2],
    num_res_units=2,
    norm="batch",
)

NN(d['image'].unsqueeze(0))
d['image'].unsqueeze(0).shape
d['image'].shape


# %% Image statistics
minpx = d["image"].min()
maxpx = d["image"].max()
print(f"Mean: {d['image'].mean()}")
print(f"Std: {d['image'].std()}")
print(f"Median: {np.median(d['image'])}")
print(f"Min: {minpx}")
print(f"Max: {maxpx}")
h = histogram1d(d["image"].flatten(), range=(minpx, maxpx), bins=256)
plt.plot(np.linspace(minpx,maxpx,256),h)
plt.show()



#%%
#NormalizeLabelsInDatasetd(keys="label", label_names=self._labels),  # Specially for missing labels
d = NormalizeIntensityd(keys="image", nonzero=True)(d)   # JH: mean centering and unit variance scaling
print(f"{d['image'].shape} => {d['label'].shape}")

plt.imshow(d["image"][0, :, :, 128], cmap="gray")
plt.show()

# %% Image statistics
minpx = d["image"].min()
maxpx = d["image"].max()
print(f"Mean: {d['image'].mean()}")
print(f"Std: {d['image'].std()}")
print(f"Median: {np.median(d['image'])}")
print(f"Min: {minpx}")
print(f"Max: {maxpx}")
h = histogram1d(d["image"].flatten(), range=(minpx, maxpx), bins=256)
plt.plot(np.linspace(minpx,maxpx,256),h)
plt.show()

# %%
d = ScaleIntensityd(keys="image")(d) # minv=-1.0, maxv=1.0),

#%% Image statistics
minpx = d["image"].min()
maxpx = d["image"].max()
print(f"Mean: {d['image'].mean()}")
print(f"Std: {d['image'].std()}")
print(f"Median: {np.median(d['image'])}")
print(f"Min: {minpx}")
print(f"Max: {maxpx}")
h = histogram1d(d["image"].flatten(), range=(minpx, maxpx), bins=256)
plt.plot(np.linspace(minpx,maxpx,256),h)
plt.show()

# %%
A = SelectItemsd(keys=("image", "label"))(d)
# %%
A["image"].shape
# %%
