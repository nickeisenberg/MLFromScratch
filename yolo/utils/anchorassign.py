import torch
from .iou import iou

class AnchorAssign:
    def __init__(self, anchors, scales, annotes):
        self.annotes = annotes
        self.anchors = anchors
        self.scales = scales
        self.anchor_assignment = {}
        self.ignore_keys = []

    def best_anchor_for_annote(self, annote, ignore_keys=[]):
        bbox = annote['bbox']
        best_iou = -1
        best_key = "0_0_0_0"
        for i, anchor in enumerate(self.anchors):
            anchor_id = i % 3
            anchor_scale = i // 3
            which_cell_row = (bbox[1] + (bbox[3] // 2)) // self.scales[anchor_scale]
            which_cell_col = (bbox[0] + (bbox[2] // 2)) // self.scales[anchor_scale]
            key = f"{anchor_id}_{anchor_scale}_{which_cell_row}_{which_cell_col}"
            if key in ignore_keys:
                continue
            _iou = iou(anchor, bbox, share_center=True)
            if _iou > best_iou:
                best_iou = _iou
                best_key = key
        if best_key not in self.anchor_assignment.keys():
            self.anchor_assignment[best_key] = (annote, best_iou)
            return None
        else:
            if best_iou > self.anchor_assignment[best_key][1]:
                replaced_annote = self.anchor_assignment[best_key][0]
                self.anchor_assignment[best_key] = (annote, best_iou) 
                self.ignore_keys.append(best_key)
                self.best_anchor_for_annote(replaced_annote, self.ignore_keys)
            else:
                self.ignore_keys.append(best_key)
                self.best_anchor_for_annote(annote, self.ignore_keys)

    def annote_loop(self):
        for annote in self.annotes:
            self.best_anchor_for_annote(annote)
            self.ignore_keys = []
