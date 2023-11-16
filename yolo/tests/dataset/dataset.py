from utils import Dataset
import os
import json
from torchvision.transforms import v2
import torch
import matplotlib.pyplot as plt
from utils import iou
import numpy as np

HOME = os.environ['HOME']
FLIRROOT = os.path.join(
    HOME, 'Datasets', 'flir'
)
TRAINROOT = os.path.join(
    FLIRROOT, 'images_thermal_train'
)
ANNOT_FILE_PATH = os.path.join(
    TRAINROOT , 'coco.json'
)
with open(ANNOT_FILE_PATH, 'r') as oj:
    annotations = json.load(oj)

annotations['annotations'][0]

ANCHORS = [ 
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], 
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], 
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], 
]

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])
dataset = Dataset(
    ANNOT_FILE_PATH, 'images', 'annotations', 'file_name', 'id', 'bbox', 
    'image_id', 'category_id', transforms=transform, fix_file_path=TRAINROOT
)

for i, (im, a) in enumerate(dataset):
    im0 = im
    im0_a = a
    if i == 2:
        print(len(im0_a))
        print(im0_a)
        break


im0_a[0]['bbox']

anks = [(0, 0, int(x[0] * im0.shape[-1]), int(x[1] * im0.shape[-2])) for X in ANCHORS for x in X]

im0.shape

plt.imshow(im0[0])
plt.show()

#--------------------------------------------------
# flir image size (512, 640)
# flir output sizes
#--------------------------------------------------
# torch.Size([1, 3, 16, 20, 15]) -- each cell represents (32 X 32) 
# torch.Size([1, 3, 32, 40, 15]) -- each cell represents (16 X 16)
# torch.Size([1, 3, 64, 80, 15]) -- each cell represents (8 X 8)
#--------------------------------------------------

scales = [32, 16, 8]

for img_id, (img, annotes) in enumerate(dataset):
    if (img_id + 1) % 250 == 0:
        print(np.round(100 * img_id / len(dataset), 3))
    ank_tracker = {}
    for annot in annotes:
        img_id = annot['image_id']
        bbox = annot['bbox']
        bbox_id = annot['id']
        score = -1
        best_key = "0_0_0_0"
        for i, ank in enumerate(anks):
            ank_id = i % 3  # ank id
            ank_scale = i // 3  # which scale
            which_cell_row = bbox[1] // scales[ank_scale]
            which_cell_col = bbox[0] // scales[ank_scale]
            key = f"{ank_id}_{ank_scale}_{which_cell_row}_{which_cell_col}"
            _score = iou(ank, bbox, share_center=True)
            if _score > score and key not in ank_tracker:
                score = _score
                best_key = key 
            elif _score > score and key in ank_tracker:
                if _score == ank_tracker[key][1]:
                    print(" ")
                    print(bbox)
                    print(bbox_id)
                    print(img_id)
                    print(_score)
                    print(ank_tracker[key])
                    print(" ")
        ank_tracker[best_key] = (bbox_id, score)
            
ank_tracker

annotations['annotations'][1384]
annotations['annotations'][1385]


