from utils import Dataset
import os
import json
from torchvision.transforms import v2
import torch
from utils import iou, BuildTarget

TRAINROOT = os.path.join(
    os.environ['HOME'], 'Datasets', 'flir', 'images_thermal_train'
)
ANNOT_FILE_PATH = os.path.join(
    TRAINROOT , 'coco.json'
)
with open(ANNOT_FILE_PATH, 'r') as oj:
    annotations = json.load(oj)

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

anchors = torch.tensor([ 
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], 
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], 
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], 
]).reshape((-1, 2))
anchors

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

scales = [32, 16, 8]
buildtarget = BuildTarget(anchors, annotes, scales, 640, 512)
buildtarget.build_targets(match_bbox_to_pred=True)

dic = buildtarget.anchor_assignment

keys = list(dic.keys())

keys[0]
decode_key(keys[0])

buildtarget.target[2][1][14][66]

decoded_annotes = {}
scales = [32, 16, 8]
thresh = .9
for i, scale_tensor in enumerate(buildtarget.target):
    scale = scales[i]
    for a, b, c, _ in zip(*torch.where(scale_tensor[..., 4: 5] > thresh)):
        torch.tensor([a, b, c])
        st = scale_tensor[a][b][c][:4]
        x = int(scale * (c + st[0]))
        y = int(scale * (b + st[1]))
        w = int(scale * st[2])
        h = int(scale * st[3])
        decoded_annotes[f"{i}_{a}_{b}_{c}"] = [x, y, w, h]

for key in keys:
    print(dic[key][0]['bbox'] == decoded_annotes[key])



