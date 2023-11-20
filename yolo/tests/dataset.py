from utils import Dataset, BuildTarget
import os
import json
from torchvision.transforms import v2
import torch
from torch.utils.data import DataLoader

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
UNSCALED_ANCHORS = torch.tensor([
    [x[0] * 640, x[1] * 512]
    for X in ANCHORS for x in X
])
SCALES = [32, 16, 8]

# Input transform
img_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

# target transform
def target_transform(annotes):
    target =  BuildTarget(
        UNSCALED_ANCHORS, annotes, SCALES
    ).build_targets(return_target=True)
    return target

dataset = Dataset(
    annot_file_path=ANNOT_FILE_PATH, 
    annot_image_key='images', 
    annot_bbox_key='annotations', 
    image_file_name='file_name', 
    image_image_id='id', 
    bbox_bbox='bbox', 
    bbox_image_id='image_id', 
    bbox_category_id='category_id', 
    img_transform=img_transform,
    target_transform=target_transform,
    fix_file_path=TRAINROOT
)

dataloader = DataLoader(dataset, 32, shuffle=False)

for batch in dataloader:
    batch[0].shape
    batch[1][0].shape
    batch[1][1].shape
    batch[1][2].shape
    break










