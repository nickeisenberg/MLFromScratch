from utils import Dataset, BuildTarget
import os
import json
from torchvision.transforms import v2
import numpy as np
from torch.utils.data import DataLoader
import torch

TRAINROOT = os.path.join(
    os.environ['HOME'], 'Datasets', 'flir', 'images_thermal_train'
)
ANNOT_FILE_PATH = os.path.join(
    TRAINROOT , 'coco.json'
)
with open(ANNOT_FILE_PATH, 'r') as oj:
    annotations = json.load(oj)


def category_mapper(annote_json):
    
    if isinstance(annote_json, str):
        with open(ANNOT_FILE_PATH, 'r') as oj:
            annotations = json.load(oj)
    elif isinstance(annote_json, dict):
        annotations = annote_json
    else:
        print("annotations wernt loaded")
        return None

    cat_ids = {}
    for annote in annotations['annotations']:
        id = annote['category_id']
        if id not in cat_ids.keys():
            # cat_ids.append(id)
            cat_ids[id] = len(cat_ids)

    return cat_ids

cat_mapper = category_mapper(annotations)
