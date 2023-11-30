from PIL import Image, ImageDraw
from torchvision.transforms import ToPILImage
import torch
import torch.nn as nn
from train_model.settings import model_inputs, num_classes, t_at
import os
import matplotlib.pyplot as plt
from model import YoloV3, YoloV3Loss
from utils import scale_anchors, BuildTarget

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
train_pth_path = modelroot + "/state_dicts/yolo_train0.pth"

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
# See the models loss on this image
#--------------------------------------------------
loss_fn = YoloV3Loss("cpu")
loss = {}
for scale_id, (p, t) in enumerate(zip(pred, target)):
    scale = scales[scale_id]
    scaled_anchors = scale_anchors(
        anchors[3 * scale_id: 3 * (scale_id + 1)], scale, 640, 512
    )
    total_loss, loss_hist = loss_fn(p, t, scaled_anchors)
    for k in loss_hist.keys():
        if k not in loss:
            loss[k] = 0
        loss[k] += loss_hist[k]
for k in loss:
    print(k, loss[k])
#--------------------------------------------------


#--------------------------------------------------
# Decode the target and the prediction and get the bounding box info
#--------------------------------------------------
target = [t[0] for t in target]
pred = [p[0] for p in pred]
image = image[0]

target_bbox = bt.decode_tuple(target, .8, 1, False)
pred_bbox = bt.decode_tuple(pred, .75, .8, True)
#--------------------------------------------------

for b in target_bbox:
    print(b['bbox'])

#--------------------------------------------------
# Plot the bounding boxes
#--------------------------------------------------

image_pil = ToPILImage()(image)

if image_pil.mode == 'L':
    rgb_img = Image.new("RGB", image_pil.size)
    rgb_img.paste(image_pil)
    image_pil = rgb_img
draw = ImageDraw.Draw(image_pil)
for bbox in pred_bbox:
    bbox = bbox['bbox']
    for i in [0]:
        bottom_right_x = bbox[0] + i + bbox[2]
        bottom_right_y = bbox[1] + i + bbox[3]
        draw.rectangle(
            (bbox[0] + i, bbox[1] + i, bottom_right_x, bottom_right_y), 
            outline ="red"
        )
image_pil.show()




