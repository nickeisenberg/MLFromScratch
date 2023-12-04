#------------------------------------------------------------------------------
# Model settings. These variable will be impoted into the model file
#------------------------------------------------------------------------------
import os
import json
import torch
from torch.optim import Adam
from torch.utils.data.dataset import Subset, random_split
from torchvision.transforms import v2
from model import YOLOv3, YoloV3Loss
from utils import Dataset, BuildTarget, AnnotationTransformer


#------------------------------------------------------------------------------
# Set the path to the data
#------------------------------------------------------------------------------
trainroot = os.path.join(
    os.environ['HOME'], 'Datasets', 'flir', 'images_thermal_train'
)
train_annote_path = os.path.join(
    trainroot , 'coco.json'
)

with open(train_annote_path, 'r') as oj:
    t_annotations = json.load(oj)

valroot = os.path.join(
    os.environ['HOME'], 'Datasets', 'flir', 'images_thermal_val'
)
val_annote_path = os.path.join(
    valroot , 'coco.json'
)

with open(val_annote_path, 'r') as oj:
    v_annotations = json.load(oj)

#------------------------------------------------------------------------------
# Transform the annotations and create the key mapper
#------------------------------------------------------------------------------
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

t_at = AnnotationTransformer(t_annotations, instructions=instructions)

v_at = AnnotationTransformer(v_annotations, instructions=instructions)

t_annotations = t_at.annotations
v_annotations = v_at.annotations

cat_map = t_at.cat_mapper

num_classes = len(cat_map)

#------------------------------------------------------------------------------
# Set the device
#------------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

#------------------------------------------------------------------------------
# Define the anchors
#------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------
# Image sizes and image scales
#------------------------------------------------------------------------------
image_size = (1, 640, 512)
scales = torch.tensor([32, 16, 8])

#------------------------------------------------------------------------------
# Set your train and validate dataset (t_dataset and v_dataset respectively)
#------------------------------------------------------------------------------
img_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

def target_transform(annotes, anchors=anchors, scales=scales):
    target =  BuildTarget(
        cat_map, anchors, scales, image_size[-1], image_size[-2]
    ).build_target(annotes, return_target=True, is_model_pred=True)
    return target

t_dataset = Dataset(
    annot_json=t_annotations,
    annot_image_key='images', 
    annot_bbox_key='annotations', 
    image_file_name='file_name', 
    image_image_id='id', 
    bbox_bbox='bbox', 
    bbox_image_id='image_id', 
    bbox_category_id='category_id', 
    img_transform=img_transform,
    target_transform=target_transform,
    fix_file_path=trainroot
)

v_dataset = Dataset(
    annot_json=v_annotations,
    annot_image_key='images', 
    annot_bbox_key='annotations', 
    image_file_name='file_name', 
    image_image_id='id', 
    bbox_bbox='bbox', 
    bbox_image_id='image_id', 
    bbox_category_id='category_id', 
    img_transform=img_transform,
    target_transform=target_transform,
    fix_file_path=valroot
)

t_dataset = Subset(t_dataset, range(6))
v_dataset = Subset(v_dataset, range(6))

batch_size = 2

notify_after = 1

#------------------------------------------------------------------------------
# Instantiate the model, set the loss and set the optimizer
#------------------------------------------------------------------------------
yoloV3 = YOLOv3(image_size[0], num_classes).to(device)
loss_fn = YoloV3Loss(device=device)
optimizer = Adam(yoloV3.parameters(), lr=.0001, weight_decay=.00001)

#------------------------------------------------------------------------------
# Put all of the model inputs into a dictionary to pass into the model
#------------------------------------------------------------------------------
model_inputs = {
    "model": yoloV3, 
    "loss_fn": loss_fn, 
    "optimizer": optimizer, 
    "t_dataset": t_dataset, 
    "v_dataset": v_dataset,
    "batch_size": batch_size, 
    "device": device, 
    "scales": scales, 
    "anchors": anchors, 
    "notify_after": notify_after,
}
