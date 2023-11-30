from utils import BuildTarget, AnnotationTransformer, scale_anchors, iou
import os
import json
import torch
import itertools
from copy import deepcopy

#--------------------------------------------------
# minimal settings
#--------------------------------------------------
annote_file_path = os.path.join(
    os.environ['HOME'], 'Datasets', 'flir', 'images_thermal_train', 'coco.json'
)
with open(annote_file_path, 'r') as oj:
    annotations = json.load(oj)

anchors = torch.tensor([
    [0.38109066, 0.53757016],
    [0.27592983, 0.2353135],
    [0.14739895, 0.37145784],
    [0.16064414, 0.13757506],
    [0.07709555, 0.21342673],
    [0.08780259, 0.07422413],
    [0.03908951, 0.11424923],
    [0.03016789, 0.05322024],
    [0.01484773, 0.02237259]
], dtype=torch.float32)

image_size = (1, 640, 512)
scales = torch.tensor([32, 16, 8])
#--------------------------------------------------

instructions = {
    'light': 'ignore',
    'sign': 'ignore',
    'hydrant': 'ignore',
    'deer': 'ignore',
    'skateboard': 'ignore',
    'train': 'ignore',
    'dog': 'ignore',
    'stroller': 'ignore',
    'scooter': 'ignore',
}
at = AnnotationTransformer(annotations, instructions)
annotations = at.annotations
cat_map = at.cat_mapper
cat_map_inv = {v: k for k, v in cat_map.items()}

image_ids = [img['id'] for img in annotations['images']]

img_id = image_ids[0]
img_annots = [
    x for x in annotations['annotations'] 
    if x['image_id'] == img_id
]

bt = BuildTarget(at.cat_mapper, anchors, img_annots, scales, 640, 512)
target = bt.build_target(return_target=True)

bt.decode_tuple(target, .8, 1, False)
