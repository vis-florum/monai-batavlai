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
import os
from typing import Any, Dict, Optional, Union

import lib.infers
import lib.trainers
from monai.networks.nets import UNet
from monai.utils import optional_import

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.utils.others.generic import download_file, strtobool

_, has_cp = optional_import("cupy")
_, has_cucim = optional_import("cucim")

logger = logging.getLogger(__name__)


class Segmentation(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        # Labels  - DON'T INCLUDE BACKGROUND LABEL
        self.labels = {
            #"heartwood": 1,
            #"sapwood": 2,
            "liveknot": 3,
            #"deadknot": 4,
            #"pith": 5,
        }
        
        # Model Files
        self.path = [
            os.path.join(self.model_dir, f"pretrained_{name}.pt"),  # pretrained
            os.path.join(self.model_dir, f"{name}.pt"),  # published
        ]

        # Download PreTrained Model
        # if strtobool(self.conf.get("use_pretrained_model", "false")):
        #     url = f"{self.conf.get('pretrained_path', self.PRE_TRAINED_PATH)}"
        #     url = f"{url}/radiology_segmentation_segresnet_multilabel.pt"
        #     #url = f"{url}/radiology_segmentation_unet_multilabel.pt"
        #     download_file(url, self.path[0])

        #self.target_spacing = (1.0, 1.0, 1.0)  # target space for image
        self.target_spacing = (1.3671875, 1.3671875, 10.0)      # JH: verify -> not functional without it
        # NB: some of them are (1.5625, 1.5625, 10.0) !!!!
        
        # Setting ROI size - This is for the image padding
        # JH: sliding window size for train and infer
        #self.roi_size = (256, 256, 128)  
        self.roi_size = (256, 256, 512)

        # Network
        self.network = UNet(
            spatial_dims=3,
            in_channels=1, #self.number_intensity_ch,
            out_channels=len(self.labels.keys()) + 1,  # All labels plus background
            channels=[16, 32, 64, 128, 256],
            strides=[2, 2, 2, 2],
            num_res_units=2,
            norm="batch",
        )

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        task: InferTask = lib.infers.Segmentation(
            path=self.path,
            network=self.network,
            roi_size=self.roi_size,
            target_spacing=self.target_spacing,
            labels=self.labels,
            preload=strtobool(self.conf.get("preload", "false")),
            config={"largest_cc": False},  # JH: do not keep largest connected component
        )
        return task


    def trainer(self) -> Optional[TrainTask]:
        output_dir = os.path.join(self.model_dir, self.name)
        load_path = self.path[0] if os.path.exists(self.path[0]) else self.path[1]

        task: TrainTask = lib.trainers.Segmentation(
            model_dir=output_dir,
            network=self.network,
            roi_size=self.roi_size,
            target_spacing=self.target_spacing,
            load_path=load_path,
            publish_path=self.path[1],
            # Training specs, NB: can be overrided in main.py !
            config={"name": "train_LPD",
                    "max_epochs": 800,
                    "train_batch_size": 2,
                    "val_batch_size": 2,
                    "multi_gpu": True,
                    #"val_split": 0.2,
                    "val_dir": self.conf.get("validation"),  # JH: External validation set
                    #"train_ds": "/media/Store-SSD/Stembank/pine/LPDsample/train/*",  # same as where server resides?
                    #"val_ds": "/media/Store-SSD/Stembank/pine/LPDsample-nrrd/val/datastore_v2.json",
                    },  # JH: added
            description="Train BATA-VLAI Segmentation Model",
            labels=self.labels,
            disable_meta_tracking=False,   # JH ported from old UNet
        )
        return task