from PIL import Image, ImageDraw
from torchvision.transforms import ToPILImage
from model import Model, YoloV3
from utils import scale_anchors, iou
from train_model.settings import t_dataset, v_dataset, scales, anchors, num_classes
import matplotlib.pyplot as plt
import torch
import numpy as np
import itertools

from utils.scale_anchors import scale_anchors

yoloV3 = YoloV3(1, scales, num_classes)

model = Model(model=yoloV3, t_dataset=t_dataset, v_dataset=v_dataset, 
              scales=scales, anchors=anchors, device="cpu")

sd = "/home/nicholas/GitRepos/ml_arcs/yolo/train_model/state_dicts/yolo_1.pth"
model.load_state_dict(sd)


#--------------------------------------------------
# add bounding boxes to images
#--------------------------------------------------
img_arr, tar = model.t_dataset[1]

pred = [_pred.detach().numpy()[0] for _pred in model.model(img_arr.unsqueeze(0))]

for scale_id, ps in enumerate(pred):
    scale = scales[scale_id].item()
    sanchors = scale_anchors(anchors[scale_id: scale_id + 3], scale, 640, 512)
    dims = itertools.product(range(ps.shape[0]), range(ps.shape[1]), range(ps.shape[2]))
    for anc, row, col in dims:
        x, y, w, h, p = ps[anc][row][col][: 5]
        if w > xxx:
            xxx = w
        x, y = np.round((x + col) * scale), np.round((y + row) * scale)
        w = np.round(np.exp(w) * sanchors[anc][0].item())
        h = np.round(np.exp(h) * sanchors[anc][1].item())
        pred[scale_id][anc][row][col][:4] = np.array([x, y, w, h])

p_thresh = {0: .6, 1: .6, 2: .99}
iou_thresh = {0: .1, 1: .1, 2: .001}
keeps = {}
for scale_id, ps in enumerate(pred):
    keeps[scale_id] = []
    probs = ps[..., 4: 5].reshape(-1)
    probs = probs[np.argsort(probs)][:: -1]
    dims = list(zip(*np.where(ps[..., 4:5] > p_thresh[scale_id])[:-1]))
    for prob in probs:
        if prob < p_thresh[scale_id]:
            break
        anc, row, col, _ = [ind[0] for ind in np.where(ps == prob)]
        bbox = torch.tensor(ps[anc][row][col][: 4])
        keeps[scale_id].append(((anc, row, col), bbox))
        for dim in dims:
            if dim == (anc, row, col):
                dims.remove((anc, row, col))
                continue
            bbox_comp = torch.tensor(ps[dim][: 4])
            score = iou(bbox, bbox_comp)
            if score > iou_thresh[scale_id]:
                dims.remove(dim)

keeps[2]

img = ToPILImage()(img_arr)

if img.mode == 'L':
    rgb_img = Image.new("RGB", img.size)
    rgb_img.paste(img)
    img = rgb_img

draw = ImageDraw.Draw(img)

for bbox in keeps[1]:
    bbox = bbox[1]
    for i in [0]:
        bottom_right_x = bbox[0] + i + bbox[2]
        bottom_right_y = bbox[1] + i + bbox[3]
        draw.rectangle(
            (bbox[0] + i, bbox[1] + i, bottom_right_x, bottom_right_y), 
            outline ="red"
        )

img.show()

