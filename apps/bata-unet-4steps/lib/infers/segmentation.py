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

from typing import Callable, Sequence

from lib.transforms.transforms import GetCentroidsd, Unpadd
from monai.inferers import Inferer, SlidingWindowInferer
from monai.inferers import SimpleInferer
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    GaussianSmoothd,
    KeepLargestConnectedComponentd,
    LoadImaged,
    SaveImaged,
    NormalizeIntensityd,
    Orientationd,
    ResizeWithPadOrCropd,
    ScaleIntensityd,
    Spacingd,
    SpatialCropd,
    SpatialPadd,
)

from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.transform.post import Restored


class Segmentation(BasicInferTask):
    """
    This provides Inference Engine for pre-trained Segmentation model.
    """

    def __init__(
        self,
        path,
        network=None,
        target_spacing=(1.0, 1.0, 1.0),
        type=InferType.SEGMENTATION,
        labels=None,
        dimension=3,
        description="A pre-trained model for volumetric (3D) Segmentation from CT image",
        **kwargs,
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            load_strict=False,
            **kwargs,
        )
        self.target_spacing = target_spacing

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        t = [
            # Ensure Structure
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys="image", device=data.get("device") if data else None),    # Ensure the input data to be a PyTorch Tensor or numpy array
            #
            # Geometry
            # Orientationd(keys=("image", "label"), axcodes="IJK"),  # JH not applicable for stembank, but make sure it is noted in starting files!
            Spacingd(keys="image", pixdim=self.target_spacing, allow_missing_keys=True), #  JH: need allow missing keys here for some reason
            #
            # Cropping and Image Sizing
            # CropForegroundd(keys=("image", "label"), source_key="image"),   # simply thresholds on >0 by default
            # CropForegroundd(keys=("image", "label"), source_key="image", margin=10, k_divisible=[self.roi_size[0], self.roi_size[1], self.roi_size[2]]),   # simply thresholds on >0 by default
            # CenterSpatialCropd(keys=("image", "label"), roi_size=(self.roi_size[0],self.roi_size[1],-1)),  # JH: crop symmetrically from middle, ignore length direction
            # SpatialPadd(keys=("image", "label"), spatial_size=self.roi_size, method="end", mode="constant", constant_values=0),
            ResizeWithPadOrCropd(keys="image", spatial_size=self.roi_size, mode="constant", method="end", constant_values=0),    # JH: Merges above two operations
            #
            # Intensity Value Changes            
            # NormalizeIntensityd(keys="image", nonzero=True),    # JH: mean centering and unit variance scaling => use ScaleIntensityd instead
            ScaleIntensityd(keys="image"), #minv=-1.0, maxv=1.0),   # automatically from 0 to 1
        ]
        return t

    def inferer(self, data=None) -> Inferer:    # JH: change to SimpleInferer??
        # return SlidingWindowInferer(
        #     roi_size=self.roi_size,
        #     #sw_batch_size=2,
        #     #overlap=0.4,
        #     #padding_mode="replicate",
        #     #mode="gaussian",
        # )
        return SimpleInferer()

    # Inverse transforms (defined inside forward methods)
    # Return one of the following.
    #         - None: Return None to disable running any inverse transforms (default behavior).
    #         - Empty: Return [] to run all applicable pre-transforms which has inverse method
    #         - list: Return list of specific pre-transforms names/classes to run inverse method
    def inverse_transforms(self, data=None):
        return []

    def post_transforms(self, data=None) -> Sequence[Callable]:
        t = [
            EnsureTyped(keys="image", device=data.get("device") if data else None),
            Activationsd(keys="pred", softmax=True),    # JH: or sigmoid?
            AsDiscreted(keys="pred", argmax=True),
            #Unpadd(keys=("pred"),ref_image="image"),   # Custom transform to restore original size -> need to to before Restored otherwise resizing takes place
            Restored(   # Restore the original spacing by resizing (!), orientation and label indices
                    keys="pred",
                    ref_image="image",
                    config_labels=self.labels,
                ),
            # NB: need to add the cropped frame back if spacing is diffeent than standard
        ]

        # if data and data.get("largest_cc", False):
        #     t.append(KeepLargestConnectedComponentd(keys="pred"))   # only keeps one!
        # t.extend(
        #     [
        #         Restored(   # Restore the original label indices
        #             keys="pred",
        #             ref_image="image",
        #             config_labels=self.labels if data.get("restore_label_idx", False) else None,
        #         ),
        #         # GetCentroidsd(keys="pred", centroids_key="centroids"),
        #     ]
        # )
        return t
