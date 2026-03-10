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

# Adapted by Johannes Huber
#
# Inference:
# python apps/bata-unet/main.py --studies /media/Store-SSD/Stembank/pine/LPDsample/test --model segmentation --test infer



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
#from monailabel.datastore.local import LocalDatastore   # JH added for datastore initialisation
#from monailabel.config import settings  # JH added for datastore initialisation
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

# bundle
from monailabel.tasks.infer.bundle import BundleInferTask
from monailabel.tasks.train.bundle import BundleTrainTask
from monailabel.utils.others.class_utils import get_class_names
from monailabel.utils.others.generic import get_bundle_models, strtobool
from monailabel.utils.others.planner import HeuristicPlanner


logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        
        ### JH addded
        modelvariant = conf.get("variant")
        validation_dir = conf.get("validation")
        ###
        
        self.model_dir = os.path.join(app_dir, "model_" + modelvariant)

        configs = {}
        for c in get_class_names(lib.configs, "TaskConfig"):
            name = c.split(".")[-2].lower() # name before .py
            configs[name] = c

        configs = {k: v for k, v in sorted(configs.items())}    # write names and locations in dict

        # Load models from app model implementation, e.g., --conf models <segmentation_spleen>
        models = conf.get("models")
        if not models:
            print("")
            print("---------------------------------------------------------------------------------------")
            print("Provide --conf models <name>")
            print("Following are the available models.  You can pass comma (,) seperated names to pass multiple")
            print(f"    all, {', '.join(configs.keys())}")
            print("---------------------------------------------------------------------------------------")
            print("")
            exit(-1)

        models = models.split(",")
        models = [m.strip() for m in models]
        invalid = [m for m in models if m != "all" and not configs.get(m)]
        if invalid:
            print("")
            print("---------------------------------------------------------------------------------------")
            print(f"Invalid Model(s) are provided: {invalid}")
            print("Following are the available models.  You can pass comma (,) seperated names to pass multiple")
            print(f"    all, {', '.join(configs.keys())}")
            print("---------------------------------------------------------------------------------------")
            print("")
            exit(-1)

        # Use Heuristic Planner to determine target spacing and spatial size based on dataset+gpu
        # JH: default values of .get() are used if the key is not found
        spatial_size = json.loads(conf.get("spatial_size", "[48, 48, 32]"))
        target_spacing = json.loads(conf.get("target_spacing", "[1.0, 1.0, 1.0]"))
        self.heuristic_planner = strtobool(conf.get("heuristic_planner", "false"))
        self.planner = HeuristicPlanner(spatial_size=spatial_size, target_spacing=target_spacing)

        # app models
        self.models: Dict[str, TaskConfig] = {}
        for n in models:
            for k, v in configs.items():
                if self.models.get(k):
                    continue
                if n == k or n == "all":
                    logger.info(f"+++ Adding Model: {k} => {v}")
                    self.models[k] = eval(f"{v}()")
                    self.models[k].init(k, self.model_dir, conf, self.planner)  # NB conf flows down to model config!
        logger.info(f"+++ Using Models: {list(self.models.keys())}")
        logger.info("+++++ MONAI Label: BATA-VLAI +++++")
        

        # Load models from bundle config files, local or released in Model-Zoo, e.g., --conf bundles <spleen_ct_segmentation>
        self.bundles = get_bundle_models(app_dir, conf, conf_key="bundles") if conf.get("bundles") else None
        

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name=f"MONAILabel - BATA-VLAI ({monailabel.__version__})",
            description="DeepLearning models for BATA-VLAI",
            version=monailabel.__version__,
        )
        

    def init_datastore(self) -> Datastore:
        datastore = super().init_datastore()
        if self.heuristic_planner:
            self.planner.run(datastore)
        return datastore


    def init_infers(self) -> Dict[str, InferTask]:
        infers: Dict[str, InferTask] = {}

        #################################################
        # Models
        #################################################
        for n, task_config in self.models.items():
            c = task_config.infer()
            c = c if isinstance(c, dict) else {n: c}
            for k, v in c.items():
                logger.info(f"+++ Adding Inferer:: {k} => {v}")
                infers[k] = v

        #################################################
        # Bundle Models
        #################################################
        if self.bundles:
            for n, b in self.bundles.items():
                i = BundleInferTask(b, self.conf)
                logger.info(f"+++ Adding Bundle Inferer:: {n} => {i}")
                infers[n] = i
        return infers

    def init_trainers(self) -> Dict[str, TrainTask]:
        trainers: Dict[str, TrainTask] = {}
        if strtobool(self.conf.get("skip_trainers", "false")):
            return trainers
        #################################################
        # Models
        #################################################
        for n, task_config in self.models.items():
            t = task_config.trainer()
            if not t:
                continue

            logger.info(f"+++ Adding Trainer:: {n} => {t}")
            trainers[n] = t

        #################################################
        # Bundle Models
        #################################################
        if self.bundles:
            for n, b in self.bundles.items():
                t = BundleTrainTask(b, self.conf)
                if not t or not t.is_valid():
                    continue

                logger.info(f"+++ Adding Bundle Trainer:: {n} => {t}")
                trainers[n] = t

        return trainers

    def init_strategies(self) -> Dict[str, Strategy]:
        strategies: Dict[str, Strategy] = {
            "random": Random(),
            "first": First(),
            "last": Last(),
        }

        if strtobool(self.conf.get("skip_strategies", "true")):
            return strategies

        for n, task_config in self.models.items():
            s = task_config.strategy()
            if not s:
                continue
            s = s if isinstance(s, dict) else {n: s}
            for k, v in s.items():
                logger.info(f"+++ Adding Strategy:: {k} => {v}")
                strategies[k] = v

        logger.info(f"Active Learning Strategies:: {list(strategies.keys())}")
        return strategies

    def init_scoring_methods(self) -> Dict[str, ScoringMethod]:
        methods: Dict[str, ScoringMethod] = {}
        # if strtobool(self.conf.get("skip_scoring", "false")):
        #     return methods

        for n, task_config in self.models.items():
            s = task_config.scoring_method()
            if not s:
                continue
            s = s if isinstance(s, dict) else {n: s}
            for k, v in s.items():
                logger.info(f"+++ Adding Scoring Method:: {k} => {v}")
                methods[k] = v

        logger.info(f"Active Learning Scoring Methods:: {list(methods.keys())}")
        return methods


"""
Example to run train/infer/batch infer/scoring task(s) locally without actually running MONAI Label Server

More about the available app methods, please check the interface monailabel/interfaces/app.py

"""

def main():
    import argparse
    import shutil
    from pathlib import Path

    from monailabel.utils.others.generic import device_list, file_ext

    os.putenv("MASTER_ADDR", "127.0.0.1")
    os.putenv("MASTER_PORT", "1234")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    # home = str(Path.home())
    # studies = f"{home}/Dataset/Radiology"
    studies = "/media/Store-SSD/Stembank/pine/server-testing"
    studies_val = "/media/Store-SSD/Stembank/pine/LPDsample/val"

    # JH note: set the default command line arguments below
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--studies", default=studies)
    parser.add_argument("-m", "--model", default="segmentation")
    parser.add_argument("-t", "--test", default="train", choices=("train", "infer", "batch_infer", "scoring"))
    parser.add_argument("-v", "--variant", default="LPDsample-CT", help="Name of the model variant")
    parser.add_argument("-i", "--inflabel", default="LPDsample-CT", help="Name of the label dir for inference")
    parser.add_argument("-x", "--validation", default=studies_val, help="Name of the validation dir")
    args = parser.parse_args()

    app_dir = os.path.dirname(__file__)
    studies = args.studies
    conf = {
        "models": args.model,
        "variant": args.variant,
        "validation": args.validation,
        "preload": "true",
    }

    app = MyApp(app_dir, studies, conf)
    
    logger.info("+++++ Staring the APP +++++")

    # Infer on Test set
    if args.test == "infer":
        sample = app.next_sample(request={"strategy": "first"})
        image_id = sample["id"]
        image_path = sample["path"]

        # Run on all devices
        for device in device_list():
            res = app.infer(
                request={"model": args.model, "image": image_id, "device": device}
            )
            # res = app.infer(
            #     request={"model": "vertebra_pipeline", "image": image_id, "device": device, "slicer": False}
            # )
            label = res["file"]
            label_json = res["params"]
            test_dir = os.path.join(args.studies, "test_labels")
            os.makedirs(test_dir, exist_ok=True)

            label_file = os.path.join(test_dir, image_id + file_ext(image_path))
            shutil.move(label, label_file)

            print(label_json)
            print(f"++++ Image File: {image_path}")
            print(f"++++ Label File: {label_file}")
            break
        return

    # Batch Infer on test set
    if args.test == "batch_infer":
        app.batch_infer(
            request={
                "model": args.model,
                "multi_gpu": True,  # JH changed
                "save_label": True,
                "label_tag": args.inflabel,
                "max_workers": 0,   # will be set automatically
                "max_batch_size": 0,    # will be set automatically, will limit number of samples looked at if > 0
            }
        )
        return
    
    if args.test == "scoring":
        app.scoring(
            request={
                "device": "cuda",
                "method": "dice",   
                "y": "final",   # dir name of ground truth
                "y_pred": args.inflabel,   # dir name of inference
            }
        )
        return

    # Train
    app.train(    # AttributeError: 'MyApp' object has no attribute '_trainers'???????
        request={
            "model": args.model,
            "dataset": "CacheDataset",  # Dataset, PersistentDataset, CacheDataset, SmartCacheDataset, https://github.com/Project-MONAI/tutorials/blob/main/acceleration/dataset_type_performance.ipynb
            # NB: Specify rest in config/segmentation.py:
        },
    )



if __name__ == "__main__":
    # Tell python to loog for libraries in these paths:
    # export PYTHONPATH=/home/aime/monai/monaivenv122:`pwd`
    # python main.py
    main()
    
