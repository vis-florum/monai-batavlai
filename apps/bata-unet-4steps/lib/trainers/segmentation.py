# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import glob
import os

import torch
from lib.transforms.transforms import NormalizeLabelsInDatasetd
from monai.handlers import TensorBoardImageHandler, from_engine
from monai.inferers import SlidingWindowInferer
from monai.inferers import SimpleInferer
from monai.losses import DiceCELoss
from monai.losses import DiceLoss   # JH added
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    CropForegroundd,
    CenterSpatialCropd,
    EnsureChannelFirstd,
    EnsureTyped,
    GaussianSmoothd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandRotated,
    RandSpatialCropd,
    ResizeWithPadOrCropd,
    ScaleIntensityd,
    SelectItemsd,
    Spacingd,
    SpatialPadd,
)

from monailabel.tasks.train.basic_train import BasicTrainTask, Context
from monailabel.tasks.train.utils import region_wise_metrics
#from monailabel.datastore.local import LocalDatastore  # JH tried

logger = logging.getLogger(__name__)


class Segmentation(BasicTrainTask):
    def __init__(   # default vals:
        self,
        model_dir,
        network,
        roi_size=(96, 96, 96),
        target_spacing=(1.0, 1.0, 1.0),
        num_samples=4,
        description="Train Segmentation model",
        **kwargs,
    ):
        self._network = network
        self.roi_size = roi_size
        self.target_spacing = target_spacing
        self.num_samples = num_samples
        super().__init__(model_dir, description, **kwargs)

    def network(self, context: Context):
        return self._network

    def optimizer(self, context: Context):
        return torch.optim.Adam(context.network.parameters(), lr=1e-4)
        #return torch.optim.AdamW(context.network.parameters(), lr=1e-4, weight_decay=1e-5)  # NB: now with decaying weights

    def loss_function(self, context: Context):
        return DiceCELoss(to_onehot_y=True, softmax=True)
        #return DiceLoss(to_onehot_y=True, softmax=True)   #JH changed, NB: try without softmax

    def lr_scheduler_handler(self, context: Context):
        return None

    def train_data_loader(self, context, num_workers=0, shuffle=False):
        return super().train_data_loader(context, num_workers, True)
    
    
    def train_pre_transforms(self, context: Context):
        return [
            # Ensure Structure
            LoadImaged(keys=("image", "label")),
            EnsureChannelFirstd(keys=("image", "label")),
            EnsureTyped(keys=("image", "label"), device=context.device),    # Ensure the input data to be a PyTorch Tensor or numpy array
            #
            # Geometry
            # Orientationd(keys=("image", "label"), axcodes="RAS"),  # JH inverts the label wrongly
            Spacingd(keys=("image", "label"), pixdim=self.target_spacing, mode=("bilinear", "nearest")), # Makes number of pixels larger if spacing is larger than expected!
            #
            # Cropping and Image Sizing
            # CropForegroundd(keys=("image", "label"), source_key="image"),   # simply thresholds on >0 by default
            # CropForegroundd(keys=("image", "label"), source_key="image", margin=10, k_divisible=[self.roi_size[0], self.roi_size[1], self.roi_size[2]]),   # simply thresholds on >0 by default
            # CenterSpatialCropd(keys=("image", "label"), roi_size=(self.roi_size[0],self.roi_size[1],-1)),  # JH: crop symmetrically from middle, ignore length direction
            # SpatialPadd(keys=("image", "label"), spatial_size=self.roi_size, method="end", mode="constant", constant_values=0),
            ResizeWithPadOrCropd(keys=("image", "label"), spatial_size=self.roi_size, mode="constant", method="end", constant_values=0),    # JH: Merges above two operations
            #
            # Intensity Value Changes            
            NormalizeLabelsInDatasetd(keys="label", label_names=self._labels),  # Specially for missing labels
            # NormalizeIntensityd(keys="image", nonzero=True),    # JH: mean centering and unit variance scaling => use ScaleIntensityd instead
            ScaleIntensityd(keys="image"), #minv=-1.0, maxv=1.0),
            # GaussianSmoothd(keys="image", sigma=0.4),
            #
            # Random Augmentations
            # RandSpatialCropd(
            #     keys=["image", "label"],
            #     roi_size=[self.roi_size[0], self.roi_size[1], self.roi_size[2]],
            #     random_size=False,
            # ),
            RandRotated(
                keys=["image", "label"],
                range_z=3.141592653589793/2,  # automatically samples from (-range_x, range_x)
                prob=0.2,
            ),
            RandFlipd(keys=("image", "label"), spatial_axis=[0], prob=0.15),
            RandFlipd(keys=("image", "label"), spatial_axis=[1], prob=0.15),
            #
            SelectItemsd(keys=("image", "label")),  #  Select only specified items from data dictionary to release memory.
                                                    # It will copy the selected key-values and construct a new dictionary.
        ]

    def train_post_transforms(self, context: Context):  # will be used as val_post_transforms automatically
        return [
            EnsureTyped(keys="pred", device=context.device),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(
                keys=("pred", "label"),
                argmax=(True, False),
                to_onehot=len(self._labels) + 1,
            ),
        ]

    def val_pre_transforms(self, context: Context):
        return [
            # Ensure Structure
            LoadImaged(keys=("image", "label")),
            EnsureChannelFirstd(keys=("image", "label")),
            EnsureTyped(keys=("image", "label"), device=context.device),    # Ensure the input data to be a PyTorch Tensor or numpy array
            #
            # Geometry
            #  Orientationd(keys=("image", "label"), axcodes="RAS"),  # JH not applicable for stembank, but make sure it is noted in starting files!
            Spacingd(keys=("image", "label"), pixdim=self.target_spacing, mode=("bilinear", "nearest")), # Makes number of pixels larger if spacing is larger than expected!
            #
            # Cropping and Image Sizing
            # CropForegroundd(keys=("image", "label"), source_key="image"),   # simply thresholds on >0 by default
            # CropForegroundd(keys=("image", "label"), source_key="image", margin=10, k_divisible=[self.roi_size[0], self.roi_size[1], self.roi_size[2]]),   # simply thresholds on >0 by default
            # CenterSpatialCropd(keys=("image", "label"), roi_size=(self.roi_size[0],self.roi_size[1],-1)),  # JH: crop symmetrically from middle, ignore length direction
            # SpatialPadd(keys=("image", "label"), spatial_size=self.roi_size, method="end", mode="constant", constant_values=0),
            ResizeWithPadOrCropd(keys=("image", "label"), spatial_size=self.roi_size, mode="constant", method="end", constant_values=0),    # JH: Merges above two operations
            #
            # Intensity Value Changes            
            NormalizeLabelsInDatasetd(keys="label", label_names=self._labels),  # Specially for missing labels
            # NormalizeIntensityd(keys="image", nonzero=True),    # JH: mean centering and unit variance scaling => use ScaleIntensityd instead
            ScaleIntensityd(keys="image"), #minv=-1.0, maxv=1.0),
            # GaussianSmoothd(keys="image", sigma=0.4),
            #
            SelectItemsd(keys=("image", "label")),  #  Select only specified items from data dictionary to release memory.
                                                    # It will copy the selected key-values and construct a new dictionary.
        ]

    # JH: Train inferer is SimplInferer by default
    def val_inferer(self, context: Context):
        # return SlidingWindowInferer(
        #     roi_size=self.roi_size,
        #     #sw_batch_size=2,
        #     #overlap=0.4,
        #     #padding_mode="replicate",
        #     #mode="gaussian"
        # )
        return SimpleInferer()

    def norm_labels(self):
        # This should be applied along with NormalizeLabelsInDatasetd transform
        new_label_nums = {}
        for idx, (key_label, val_label) in enumerate(self._labels.items(), start=1):
            if key_label != "background":
                new_label_nums[key_label] = idx
            if key_label == "background":
                new_label_nums["background"] = 0
        return new_label_nums

    def train_key_metric(self, context: Context):
        return region_wise_metrics(self.norm_labels(), "train_mean_dice", "train")

    def val_key_metric(self, context: Context):
        return region_wise_metrics(self.norm_labels(), "val_mean_dice", "val")

    def train_handlers(self, context: Context):
        handlers = super().train_handlers(context)
        if context.local_rank == 0:
            handlers.append(
                TensorBoardImageHandler(        # TENSORBOARD communication
                    log_dir=context.events_dir,
                    batch_transform=from_engine(["image", "label"]),
                    output_transform=from_engine(["pred"]),
                    interval=5,  # plot content from engine.state every N epochs or every N iterations
                    epoch_level=True,
                )
            )
        return handlers
    
    # JH added to monitor validation loss
    def val_additional_metrics(self, context: Context):
        from ignite.metrics import Loss
        return {"val_loss": Loss(loss_fn=DiceCELoss(to_onehot_y=False, softmax=True)    
                                ,output_transform=from_engine(["pred", "label"]))}
        # to_onehot_y=True gives error:
        #  File "/home/aime/monai/monaivenv122/lib/python3.8/site-packages/monai/networks/utils.py", line 187, in one_hot
        # raise AssertionError("labels should have a channel with length equal to one.")
        # AssertionError: labels should have a channel with length equal to one.
    
    # doesnt work:
    # def val_additional_metrics(self, context: Context):
    #     from ignite.metrics import Loss
    #     return {"val_loss": Loss(loss_fn=loss_function,
    #                             output_transform=from_engine(["pred", "label"]))}
   
    
    # JH: User-defined training and valdiation data
    def partition_datalist(self, context: Context, shuffle=False):
        train_d = context.datalist
        val_dir = context.request.get("val_dir")
        if not val_dir:
            raise ValueError("Validation directory is not provided")
                
        patterns = ['*.nrrd', '*.nii', '*.nii.gz']  # allwed file endings
        val_images = []
        val_labels = []
        for pattern in patterns:
            val_images.extend(glob.glob(os.path.join(val_dir, pattern)))
            val_labels.extend(glob.glob(os.path.join(val_dir, "labels/final/", pattern)))
        val_d = [{"image": image_name, "label": label_name} for image_name, label_name in zip(val_images, val_labels)]

        # old (does not work for symbolic links):
        # from pathlib import Path        
        # train_dir = Path(train_d[0]["image"]).parent   # os.path.split(train_d[0]["image"])[0]
        # val_dir = os.path.join(Path(train_dir).parent, "val")

        if context.local_rank == 0:
            logger.info(f"Total Records for Training: {len(train_d)}")
            logger.info(f"Total Records for Validation: {len(val_d)}")

        return train_d, val_d
