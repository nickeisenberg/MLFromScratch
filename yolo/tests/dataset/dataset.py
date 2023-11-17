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

anchors = [
    (0, 0, int(x[0] * 640), int(x[1] * 512)) 
    for X in ANCHORS for x in X
]

scales = [32, 16, 8]

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
                look_here.append(img_id)
        ank_tracker[best_key] = (annot, score)
    all_ims_nr[img_id] = ank_tracker
    if img_id == 500:
        break

class AnchorAssign:
    def __init__(self, anchors, scales, annotes):
        self.annotes = annotes
        self.anchors = anchors
        self.scales = scales
        self.anchor_assignment = {}
        self.ignore_keys = []

    @staticmethod
    def iou(box1, box2, share_center=False):
        """
        Parameters
        ----------
        box1: torch.Tensor
            Iterable of format [bx, by, bw, bh] where bx and by are the coords of
            the top left of the bounding box and bw and bh are the width and
            height
        box2: same as box1
        pred: boolean default = False
            If False, then the assumption is made that the boxes share the same
            center.
        """
        ep = 1e-6
    
        if share_center:
            box1_a = box1[2] * box1[3]
            box2_a = box2[2] * box2[3]
            intersection_a = min(box1[2], box2[2]) * min(box1[3], box2[3])
            union_a = box1_a + box2_a - intersection_a
            return intersection_a / union_a
        
        else:
            len_x = torch.max(
                torch.sub(
                    torch.min(box1[0] + box1[2], box2[0] + box2[2]),
                    torch.max(box1[0], box2[0])
                ),
                torch.Tensor([0])
            )
            len_y = torch.max(
                torch.sub(
                    torch.min(box1[1] + box1[3], box2[1] + box2[3]),
                    torch.max(box1[1], box2[1])
                ),
                torch.Tensor([0])
            )
    
            box1_a = box1[2] * box1[3]
            box2_a = box2[2] * box2[3]
    
            intersection_a = len_x * len_y
    
            union_a = box1_a + box2_a - intersection_a + ep
    
            return intersection_a / union_a

    def best_anchor_for_annote(self, annote, ignore_keys=[]):
        bbox = annote['bbox']
        score = -1
        best_key = "0_0_0_0"
        for i, anchor in enumerate(self.anchors):
            anchor_id = i % 3
            anchor_scale = i // 3
            which_cell_row = (bbox[1] + (bbox[3] // 2)) // self.scales[anchor_scale]
            which_cell_col = (bbox[0] + (bbox[2] // 2)) // self.scales[anchor_scale]
            key = f"{anchor_id}_{anchor_scale}_{which_cell_row}_{which_cell_col}"
            if key in ignore_keys:
                continue
            _score = self.iou(anchor, bbox, share_center=True)
            if _score > score and key not in self.anchor_assignment:
                score = _score
                best_key = key 
            elif _score > score and key in self.anchor_assignment:
                score = _score
                replaced_annote = self.anchor_assignment[key][0]
                best_key = key
                self.ignore_keys.append(best_key)
                self.best_anchor_for_annote(replaced_annote, self.ignore_keys)
        self.anchor_assignment[best_key] = (annote, score)
        return None

    def annote_loop(self):
        for annote in self.annotes:
            self.best_anchor_for_annote(annote)
            self.ignore_keys = []

all_ims = {}
for img_id, (img, annotes) in enumerate(dataset):
    if (img_id + 1) % 250 == 0:
        print(np.round(100 * img_id / len(dataset), 3))
    assign_anchors = AnchorAssign(anchors, scales, annotes)
    assign_anchors.annote_loop()
    anc_assign = assign_anchors.anchor_assignment
    all_ims[img_id] = anc_assign
    if img_id == 500:
        break

look_here[3]

yes = 0
for i in range(len(all_ims)):
    sum0 = 0
    sum1 = 0
    for key0, key1 in zip(all_ims[i], all_ims_nr[i]):
        # print(key0, all_ims[18][key0][0]['id'])
        # print(key1, all_ims_nr[18][key1][0]['id'])
        sum0 += all_ims[i][key0][1]
        sum1 += all_ims_nr[i][key1][1]
    if sum0 > sum1:
        yes += 1
yes / len(all_ims)




