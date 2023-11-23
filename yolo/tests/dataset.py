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

anchors = torch.tensor([ 
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], 
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], 
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], 
]).reshape((-1, 2))
anchors

# Input transform
img_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

# target transform
scales = [32, 16, 8]



def target_transform(annotes, anchors=anchors, scales=scales):
    target =  BuildTarget(
        anchors, annotes, scales, 640, 512
    ).build_target(return_target=True)
    return target

dataset = Dataset(
    annot_json=ANNOT_FILE_PATH, 
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

dataset2 = Dataset(
    annot_json=annotations,
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

for k, (tup, tup2) in enumerate(zip(dataset, dataset2)):
    if k % 100 == 0:
        print(k)
    im = (tup[0] != tup2[0]).sum()
    tars = torch.tensor([0])
    for i in range(3):
        tars += ((tup[1][i] != tup2[1][i]).sum())
    total = im + tars
    if total.item() != 1:
        print("something is wrong")

dataloader = DataLoader(dataset, 32, shuffle=False)

for i, batch in enumerate(dataloader):
    print(i)
    batch[0].shape
    batch[1][0].shape
    batch[1][1].shape
    batch[1][2].shape
    if i == 2:
        break
