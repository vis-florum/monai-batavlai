#%% Import App and set Study

import sys
import os
import matplotlib.pyplot as plt
import importlib    # For updating modules
import numpy as np

sys.path.append("/home/aime/monai/apps/bata-unet")
import main
importlib.reload(main)
from main import MyApp

from pathlib import Path
from monailabel.utils.others.generic import device_list, file_ext

studies = "/media/Store-SSD/Stembank/pine/server-testing"
app_dir = "/home/aime/monai/apps/bata-unet"
conf = {
        "models": "segmentation",
        "preload": "true",
    }

app = MyApp(app_dir, studies, conf)


os.putenv("MASTER_ADDR", "127.0.0.1")
os.putenv("MASTER_PORT", "1234")

request={
            "model": "segmentation",
            "max_epochs": 2,     # JH changed <---- not followed by slicer
            "dataset": "Dataset",  # PersistentDataset, CacheDataset
            "train_batch_size": 1,  # Logs can be differently long??
            "val_batch_size": 1,
            "multi_gpu": False,  # JH changed
            "val_split": 0.1,
        }

app.train(request=request)