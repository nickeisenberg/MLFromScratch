from utils import BuildTarget, AnnotationTransformer, scale_anchors
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

at = AnnotationTransformer(annotations)

image_ids = [img['id'] for img in at.annotations['images']]

fails = 0
fail_list = []
for ii, img_id in enumerate(image_ids):
    if ii % 20 == 0:
        print(100 * ii / len(image_ids))

    annotes = [
        x for x in at.annotations['annotations'] if x['image_id'] == img_id
    ]
    bt = BuildTarget(at.cat_mapper, anchors, annotes, scales, 640, 512)
    bt.build_target()
    
    target = bt.target
    for scale_id, ts in enumerate(target):
        scale = scales[scale_id]
        s_ancs = scale_anchors(
            anchors[3 * scale_id: 3 * (scale_id + 1)], scales[scale_id], 640, 512
        )
        dims = itertools.product(range(ts.shape[0]), range(ts.shape[1]), range(ts.shape[2]))
        for dim in dims:
            if target[scale_id][dim][4] == 1:
                x, y, w, h = target[scale_id][dim][: 4]
                x, y = (x + dim[2]) * scale, (y + dim[1]) * scale
                w, h = w * scales[scale_id], h * scales[scale_id]
                target[scale_id][dim][:4] = torch.tensor([x, y, w, h])
    
    p_thresh = {0: .5, 1: .5, 2: .5}
    keeps = {}
    for scale_id, ts in enumerate(target):
        keeps[scale_id] = []
        probs = ts[..., 4: 5].reshape(-1)
        probs = probs[torch.argsort(probs, descending=True)]
        dims = list(zip(*torch.where(ts[..., 4:5] > p_thresh[scale_id])[:-1]))
        for dim in dims:
            keeps[scale_id].append(
                (dim, ts[dim])
            )

    annote_list = deepcopy(annotes)
    
    count = 0
    for annote in annotes:
        bbox = torch.tensor(annote['bbox'])
        cat_id = annote['category_id']
        mapped_cat_id = torch.tensor(at.cat_mapper[cat_id])
        for k in keeps.keys():
            for info in keeps[k]:
                try:
                    bbox_bool = (bbox == info[1][: 4]).sum()
                    id_bool = (mapped_cat_id == info[1][-1]).sum()
                    if bbox_bool == 4 and id_bool == 1:
                        annote_list.remove(annote)
                        count += 1
                except:
                    print(img_id)
                    fail_list.append(img_id)
    
    if count == len(annotes) and len(annote_list) == 0: 
        continue
    else:
        print(f"fail {img_id}")
        fails += 1
