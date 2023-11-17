from utils import Dataset
import os
import json
from torchvision.transforms import v2
import torch
import matplotlib.pyplot as plt
from utils import iou, BuildTarget
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

#--------------------------------------------------
# There are 80 categories
#--------------------------------------------------
print([x['id'] for x in annotations['categories']])
#--------------------------------------------------

#--------------------------------------------------
# image_id starts at 0 while annotiation id starts at 1
#--------------------------------------------------
torch.Tensor(annotations['annotations'][0]['bbox'])
torch.Tensor([annotations['annotations'][0]['category_id']])

#--------------------------------------------------

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
    annot_file_path=ANNOT_FILE_PATH, 
    annot_image_key='images', 
    annot_bbox_key='annotations', 
    image_file_name='file_name', 
    image_image_id='id', 
    bbox_bbox='bbox', 
    bbox_image_id='image_id', 
    bbox_category_id='category_id', 
    transforms=transform, 
    fix_file_path=TRAINROOT
)

for im, ann in dataset:
    print(len(ann))
    break

#--------------------------------------------------
# flir image size (512, 640)
# flir output sizes
#--------------------------------------------------
# torch.Size([1, 3, 16, 20, 15]) -- each cell represents (32 X 32) 
# torch.Size([1, 3, 32, 40, 15]) -- each cell represents (16 X 16)
# torch.Size([1, 3, 64, 80, 15]) -- each cell represents (8 X 8)
#--------------------------------------------------

#--------------------------------------------------
# It will help to put the anchors in the following shape to match the
# shape of the bounding boxes
#--------------------------------------------------
anchors = [
    (0, 0, int(x[0] * 640), int(x[1] * 512)) 
    for X in ANCHORS for x in X
]

scales = [32, 16, 8]

#--------------------------------------------------
# This following for-loop assigns the anchors on a "first-come-first-serve"
# basis. This means that if a latter attotation has a higher iou for a 
# partcular anchor, scale and cell but this anchor_scale_cell was already 
# assigned eariler in the for-loop then this higher iou score is NOT swapped
# out and is assiged to the nxt highest iou score.
#--------------------------------------------------
all_ims_nr = {}
look_here = []
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
        for i, ank in enumerate(anchors):
            ank_id = i % 3  # ank id
            ank_scale = i // 3  # which scale
            which_cell_row = (bbox[1] + (bbox[3] // 2)) // scales[ank_scale]
            which_cell_col = (bbox[0] + (bbox[2] // 2)) // scales[ank_scale]
            key = f"{ank_id}_{ank_scale}_{which_cell_row}_{which_cell_col}"
            _score = iou(ank, bbox, share_center=True)
            if _score > score and key not in ank_tracker:
                score = _score
                best_key = key 
            elif _score > score and key in ank_tracker:
                if _score > ank_tracker[key][1]:
                    look_here.append(img_id)
        ank_tracker[best_key] = (annot, score)
    all_ims_nr[img_id] = ank_tracker
    if img_id == 100:
        break
#--------------------------------------------------

#--------------------------------------------------
# The BuildTarget assign class uses recursion in order to replace
# achoirs with higher scoring bounded boxes.
#--------------------------------------------------
all_ims = {}
all_ims_targets = {}
for img_id, (img, annotes) in enumerate(dataset):
    if (img_id + 1) % 250 == 0:
        print(np.round(100 * img_id / len(dataset), 3))
    anchor_assign = BuildTarget(anchors, scales, annotes)
    anchor_assign.build_targets()
    all_ims[img_id] = anchor_assign.anchor_assignment
    all_ims_targets[img_id] = anchor_assign.target
    if img_id == 0:
        break
#--------------------------------------------------

#--------------------------------------------------
# We can parse the bouding boxes and see which anchor is assigned to it.
#--------------------------------------------------
for key, (annote, score) in all_ims[0].items():
    id = key.split("_")
    scale_id = id[0]
    anchor_id = id[1]
    cell_row = id[2]
    cell_col = id[3]
    which_anchor = f"scale{scale_id}_anchor{anchor_id}_row{cell_row}_col{cell_col}"
    print(which_anchor, annote['id'], annote['category_id'], annote['bbox'])

all_ims_targets[0][2][0, 1, 17, 67]

#--------------------------------------------------


#--------------------------------------------------
# As we can see below, the for the first 500 images, if we use BuildTarget,
# then our total iou_score per image is greater than or equal to the 
# first-come-fist-serve assigne 96% of the time.
# The reason why the sum may be lower is that when you replace a bbox's anchor,
# it may get reassiged with with anoter anchor and the resulting sum of this
# swap is less than if you were to do first-come-first-serve.
#--------------------------------------------------
yes = 0
for i in range(len(all_ims)):
    sum0 = 0
    sum1 = 0
    for key0, key1 in zip(all_ims[i], all_ims_nr[i]):
        # print(key0, all_ims[18][key0][0]['id'])
        # print(key1, all_ims_nr[18][key1][0]['id'])
        sum0 += all_ims[i][key0][1]
        sum1 += all_ims_nr[i][key1][1]
    if sum0 >= sum1:
        yes += 1
    else:
        print(i)
print(yes / len(all_ims))
#--------------------------------------------------

#--------------------------------------------------
# See below how 4541 replaced 4540 and 4540's new score was lower than
# the original.
#--------------------------------------------------
i = 221
for key0, key1 in zip(all_ims[i], all_ims_nr[i]):
    print(all_ims[i][key0][0]['id'], all_ims_nr[i][key1][0]['id'])
    if all_ims[i][key0][0]['id'] != all_ims_nr[i][key1][0]['id']:
        print(all_ims[i][key0][1], all_ims_nr[i][key1][1])
#-------------
