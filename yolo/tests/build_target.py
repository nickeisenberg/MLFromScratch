from utils import Dataset
import os
import json
from torchvision.transforms import v2
import torch
from utils import iou, BuildTarget

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

img_transform = v2.Compose([
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
    img_transform=img_transform, 
    fix_file_path=TRAINROOT
)

grid_cell_scales = [32, 16, 8]
grid_cell_scale = 1
scaled_anchors = torch.tensor([
    (
        x[0] * 640 / min(grid_cell_scale, 640) , 
        x[1] * 512 / min(grid_cell_scale, 512)
    ) 
    for X in ANCHORS for x in X
])
print(scaled_anchors)

image = torch.tensor(0)
annotes = []
for _image, _annotes in dataset:
    image = _image
    annotes = _annotes
    break
    
def decode_key(key):
    id = key.split("_")
    scale_id = id[0]
    anchor_id = id[1]
    cell_row = id[2]
    cell_col = id[3]
    which_anchor = f"scale{scale_id}_anchor{anchor_id}_row{cell_row}_col{cell_col}"
    print(which_anchor)

buildtarget = BuildTarget(scaled_anchors, annotes, grid_cell_scales)
buildtarget.build_targets(match_bbox_to_pred=True)

key = list(buildtarget.anchor_assignment.keys())[0]
decode_key(key)

buildtarget.anchor_assignment[key]
buildtarget.target[1][1][13][3]
