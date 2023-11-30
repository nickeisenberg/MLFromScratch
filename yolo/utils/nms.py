import torch
from utils import iou

def nms_from_tuple(tup, p_thresh, iou_thresh):

    for scale_id, (t, pt, it) in enumerate(zip(tup, p_thresh, iou_thresh)):

        dims = list(zip(*torch.where(t[..., 4:5] > p_thresh[scale_id])[:-1]))

