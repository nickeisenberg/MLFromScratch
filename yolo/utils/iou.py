import numpy as np
import torch

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
        box1_a = box1[..., -2] * box1[..., -1]
        box2_a = box2[..., -2] * box2[..., -1]
        intersection_a = min(box1[..., -2], box2[..., -2]) * min(box1[..., -1], box2[..., -1])
        union_a = box1_a + box2_a - intersection_a
        return intersection_a / union_a
    
    else:
        len_x = torch.max(
            torch.sub(
                torch.min(
                    box1[..., 0: 1] + box1[..., 2: 3], 
                    box2[..., 0: 1] + box2[..., 2: 3]
                ),
                torch.max(box1[..., 0: 1], box2[..., 0: 1])
            ),
            torch.Tensor([[0]])
        )
        len_y = torch.max(
            torch.sub(
                torch.min(
                    box1[..., 1: 2] + box1[..., 3: 4], 
                    box2[..., 1: 2] + box2[..., 3: 4]
                ),
                torch.max(box1[..., 1: 2], box2[..., 1: 2])
            ),
            torch.Tensor([[0]])
        )

        box1_a = box1[..., 2: 3] * box1[..., 3: 4]
        box2_a = box2[..., 2: 3] * box2[..., 3: 4]

        intersection_a = len_x * len_y

        union_a = box1_a + box2_a - intersection_a + ep

        return intersection_a / union_a

