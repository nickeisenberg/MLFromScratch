from utils import Dataset, BuildTarget, AnnotationTransformer
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

at = AnnotationTransformer(annotations)

cat_map = at.cat_mapper

anchors = torch.tensor([ 
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], 
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], 
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], 
]).reshape((-1, 2))

# Input transform
img_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

# target transform
scales = [32, 16, 8]

annotes = [x for x in annotations['annotations'] if x['image_id'] == 0]

target =  BuildTarget(
    cat_map, anchors, annotes, scales, 640, 512
)
target.build_target(return_target=False, is_model_pred=True)

target.target[1][..., 3:4].reshape(-1).max()

torch.log(1e-6 + target.target[2][..., 3:4].reshape(-1).max())
torch.log(1e-6 + target.target[2][..., 3:4].reshape(-1).min())


for key in target.anchor_assignment.keys():
    key = [int(x) for x in key.split("_")]
    s, a, r, c = key
    print(target.target[s][a][r][c][5])

for an in annotes:
    print(cat_map[an['category_id']])
