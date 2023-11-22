#------------------------------------------------------------------------------
# Model settings. These variable will be impoted into the model file
#------------------------------------------------------------------------------
import os
import torch
from torch.optim import Adam
from torch.utils.data.dataset import random_split, Subset
from torchvision.transforms import v2
from model import Model, YoloV3, YoloV3Loss
from utils import Dataset, BuildTarget

#------------------------------------------------------------------------------
# Set the path to the data
#------------------------------------------------------------------------------
trainroot = os.path.join(
    os.environ['HOME'], 'Datasets', 'flir', 'images_thermal_train'
)
annot_file_path = os.path.join(
    trainroot , 'coco.json'
)

#------------------------------------------------------------------------------
# Number of classes
#------------------------------------------------------------------------------
num_classes = 80

#------------------------------------------------------------------------------
# Set the device
#------------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

#------------------------------------------------------------------------------
# Define the anchors
#------------------------------------------------------------------------------
anchors = torch.tensor([ 
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], 
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], 
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], 
]).reshape((-1, 2))

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
        anchors, annotes, scales, image_size[-1], image_size[-2]
    ).build_targets(return_target=True)
    return target

dataset = Dataset(
    annot_file_path=annot_file_path, 
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

t_dataset = Subset(dataset, range(500))

v_dataset = Subset(dataset, range(500, 550))

batch_size = 4

notify_after = 1

#------------------------------------------------------------------------------
# Instantiate the model, set the loss and set the optimizer
#------------------------------------------------------------------------------
yoloV3 = YoloV3(image_size[0], scales, num_classes).to(device)
loss_fn = YoloV3Loss(device=device)
optimizer = Adam(yoloV3.parameters(), lr=1e-5)
