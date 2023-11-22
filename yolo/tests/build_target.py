from utils import Dataset
import os
import json
from torchvision.transforms import v2
import torch
from utils import iou, BuildTarget

#--------------------------------------------------
# Some setting that need to be set
#--------------------------------------------------
trainroot = os.path.join(
    os.environ['HOME'], 'Datasets', 'flir', 'images_thermal_train'
)
annote_file_path = os.path.join(
    trainroot , 'coco.json'
)
with open(annote_file_path, 'r') as oj:
    annotations = json.load(oj)

scales = [32, 16, 8]

anchors = torch.tensor([ 
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], 
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], 
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], 
]).reshape((-1, 2))

#--------------------------------------------------
# create the dataset and get an image and its annotations for testing
#--------------------------------------------------
img_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])
dataset = Dataset(
    annot_file_path=annote_file_path, 
    annot_image_key='images', 
    annot_bbox_key='annotations', 
    image_file_name='file_name', 
    image_image_id='id', 
    bbox_bbox='bbox', 
    bbox_image_id='image_id', 
    bbox_category_id='category_id', 
    img_transform=img_transform, 
    fix_file_path=trainroot
)

image, annotes = dataset.__getitem__(0)
#--------------------------------------------------

#--------------------------------------------------
# Human readable
#--------------------------------------------------
buildtarget = BuildTarget(anchors, annotes, scales, image.shape[-1], image.shape[-2])
buildtarget.build_targets()
anc_assign = buildtarget.anchor_assignment
target = buildtarget.target

#--------------------------------------------------
# Model Ouput
#--------------------------------------------------
buildtarget_m = BuildTarget(anchors, annotes, scales, image.shape[-1], image.shape[-2])
buildtarget_m.build_targets(match_bbox_to_pred=True)
anc_assign_m = buildtarget_m.anchor_assignment
target_m = buildtarget_m.target

#--------------------------------------------------
# See if the keys match
#--------------------------------------------------
for scale, scale_ten, scale_ten_m in zip(scales, target, target_m):
    tup = torch.where(scale_ten[..., 4] == 1)
    tup_m = torch.where(scale_ten_m[..., 4] == 1)
    try:
        for x, y in zip(tup, tup_m):
            print(x == y)
    except:
        print("fail")
        print(tup)
        print(tup_m)

#--------------------------------------------------
# reconstruct human readable and see if it matches
#--------------------------------------------------
for scale, scale_ten, scale_ten_m in zip(scales, target, target_m):
    tup = torch.where(scale_ten[..., 4] == 1)
    tup_m = torch.where(scale_ten_m[..., 4] == 1)
    for (anc_id, row, col), (anc_id_m, row_m, col_m) in zip(zip(*tup), zip(*tup_m)):
        bbox = scale_ten[anc_id][row][col]
        bbox_m = scale_ten_m[anc_id_m][row_m][col_m]
        x_recon = (col_m + bbox_m[0]) * scale
        y_recon = (row_m + bbox_m[1]) * scale
        w_recon = bbox_m[2] * scale
        h_recon = bbox_m[3] * scale
        bbox_recon = torch.hstack(
            (
                torch.tensor([x_recon, y_recon, w_recon, h_recon]),
                bbox_m[4:]
            )
        )
        print(bbox_recon == bbox)
