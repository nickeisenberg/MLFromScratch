from utils import Dataset
import os
import json
from torchvision.transforms import v2
import torch
import matplotlib.pyplot as plt
from utils import iou, BuildTarget
import numpy as np

HOME = os.environ['HOME']
TRAINROOT = os.path.join(
    HOME, 'Datasets', 'flir', 'images_thermal_train'
)
ANNOT_FILE_PATH = os.path.join(
    TRAINROOT , 'coco.json'
)
with open(ANNOT_FILE_PATH, 'r') as oj:
    annotations = json.load(oj)

ANCHORS = [ 
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], 
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], 
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], 
]


scales = [32, 16, 8]

scale = 1
scaled_anchors = torch.Tensor([
    (x[0] * 640 / min(scale, 640) , x[1] * 512 / min(scale, 512)) 
    for X in ANCHORS for x in X
])
scaled_anchors = scaled_anchors.reshape((3, 3, 2))
scaled_anchors

scaled_anchors[0].reshape((1, 3, 1, 1, 2))

pred = torch.randn((32, 3, 16, 20, 85))

pred[..., 2:4] * scaled_anchors[0].reshape((1, 3, 1, 1, 2))


pred[..., 2:4].shape

