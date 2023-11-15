import numpy as np
import torch

def _iou(box1, box2, is_pred=True): 
    """
    https://www.geeksforgeeks.org/yolov3-from-scratch-using-pytorch/
    """
	if is_pred: 
		# IoU score for prediction and label 
		# box1 (prediction) and box2 (label) are both in [x, y, width, height] format 
		
		# Box coordinates of prediction 
		b1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
		b1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
		b1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
		b1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2

		# Box coordinates of ground truth 
		b2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
		b2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
		b2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
		b2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2

		# Get the coordinates of the intersection rectangle 
		x1 = torch.max(b1_x1, b2_x1) 
		y1 = torch.max(b1_y1, b2_y1) 
		x2 = torch.min(b1_x2, b2_x2) 
		y2 = torch.min(b1_y2, b2_y2) 
		# Make sure the intersection is at least 0 
		intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0) 

		# Calculate the union area 
		box1_area = abs((b1_x2 - b1_x1) * (b1_y2 - b1_y1)) 
		box2_area = abs((b2_x2 - b2_x1) * (b2_y2 - b2_y1)) 
		union = box1_area + box2_area - intersection 

		# Calculate the IoU score 
		epsilon = 1e-6
		iou_score = intersection / (union + epsilon) 

		# Return IoU score 
		return iou_score 
	
	else:
        #--------------------------------------------------
        # Nick: There a typo online from the link above. Below is correct
        # before the 2 and 3 was 0 and 1 which is incorrect.
        #--------------------------------------------------
		# IoU score based on width and height of bounding boxes 
		
		# Calculate intersection area 
		intersection_area = torch.multiply(
            torch.min(box1[..., 2], box2[..., 2]), 
            torch.min(box1[..., 3], box2[..., 3])
        )

		# Calculate union area 
		box1_area = box1[..., 2] * box1[..., 3] 
		box2_area = box2[..., 2] * box2[..., 3] 
		union_area = box1_area + box2_area - intersection_area 

		# Calculate IoU score 
		iou_score = intersection_area / union_area 

		# Return IoU score 
		return iou_score

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

b1 = torch.Tensor([2, 2, 5, 5])
b2 = torch.Tensor([3, 3, 5, 5])

b1[..., 1]
b2[..., 2]

iou(b1, b2, True)

_iou(b1, b2, False)
