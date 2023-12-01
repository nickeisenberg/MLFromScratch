from PIL import Image, ImageDraw
from torchvision.transforms import ToPILImage
import torch
import torch.nn as nn
from train_model.settings import model_inputs, num_classes, t_at
import os
import matplotlib.pyplot as plt
from model import YoloV3, YoloV3Loss, Model
from utils import scale_anchors, BuildTarget
import torch.nn as nn

#--------------------------------------------------
# Instantate the model and BuildTarget class
#--------------------------------------------------
scales = model_inputs['scales']
anchors = model_inputs['anchors']
cat_map = t_at.cat_mapper

bt = BuildTarget(cat_map, anchors, scales, 640, 512)

yoloV3 = YoloV3(1, scales, num_classes)
#--------------------------------------------------

#--------------------------------------------------
# Load model weights
#--------------------------------------------------
modelroot = f"{os.environ['HOME']}/GitRepos/ml_arcs/yolo/train_model"
train_pth_path = modelroot + "/state_dicts/yolo_train1.pth"

yoloV3.load_state_dict(torch.load(train_pth_path))
#--------------------------------------------------

#--------------------------------------------------
# Get a training image and target and prediction
#--------------------------------------------------
image, target = model_inputs['t_dataset'][1]
image = image.unsqueeze(0)
target = [t.unsqueeze(0) for t in target]
pred = [p for p in yoloV3(image)]
#--------------------------------------------------

#--------------------------------------------------
# Decode the target and the prediction and get the bounding box info
#--------------------------------------------------
target = [t[0] for t in target]
pred = [p[0] for p in pred]
image = image[0]

target_bbox = bt.decode_tuple(target, .8, 1, False)[0]
pred_bbox, pred_bbox_all = bt.decode_tuple(pred, .75, .8, True)
#--------------------------------------------------

len(target_bbox)

len(pred_bbox)

#--------------------------------------------------
# Plot the bounding boxes
#--------------------------------------------------

which_bbox = pred_bbox
which_image = image

image_pil = ToPILImage()(which_image)
if image_pil.mode == 'L':
    rgb_img = Image.new("RGB", image_pil.size)
    rgb_img.paste(image_pil)
    image_pil = rgb_img
draw = ImageDraw.Draw(image_pil)
for bbox in which_bbox:
    bbox = bbox['bbox']
    for i in [0]:
        bottom_right_x = bbox[0] + i + bbox[2]
        bottom_right_y = bbox[1] + i + bbox[3]
        draw.rectangle(
            (bbox[0] + i, bbox[1] + i, bottom_right_x, bottom_right_y), 
            outline ="red"
        )
image_pil.show()
