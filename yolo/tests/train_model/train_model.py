import os
import torch
from torch.optim import Adam
from torch.utils.data.dataset import random_split
from torchvision.transforms import v2
from model import Model, YoloV3, YoloV3Loss
from utils import Dataset, BuildTarget
from tests.train_model.settings import *

#------------------------------------------------------------------------------
# Model settings
#------------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

TRAINROOT = os.path.join(
    os.environ['HOME'], 'Datasets', 'flir', 'images_thermal_train'
)
ANNOT_FILE_PATH = os.path.join(
    TRAINROOT , 'coco.json'
)

image_size = (1, 640, 512)
scales = torch.tensor([32, 16, 8])
num_classes = 80

anchors = torch.tensor([ 
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], 
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], 
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], 
]).reshape((-1, 2))

img_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

def target_transform(annotes, anchors=anchors, scales=scales):
    target =  BuildTarget(
        anchors, annotes, scales, 640, 512
    ).build_targets(return_target=True)
    return target

dataset = Dataset(
    annot_file_path=ANNOT_FILE_PATH, 
    annot_image_key='images', 
    annot_bbox_key='annotations', 
    image_file_name='file_name', 
    image_image_id='id', 
    bbox_bbox='bbox', 
    bbox_image_id='image_id', 
    bbox_category_id='category_id', 
    img_transform=img_transform,
    target_transform=target_transform,
    fix_file_path=TRAINROOT
)


t_dataset, v_dataset, _ = random_split(dataset, [.02, .02, .96])

yoloV3 = YoloV3(image_size, scales, num_classes).to(device)
loss_fn = YoloV3Loss()
optimizer = Adam(yoloV3.parameters(), lr=1e-5)
#------------------------------------------------------------------------------

yoloV3model = Model(
    yoloV3, loss_fn, optimizer, dataset, (.8, .2), 
    4, device, scales, anchors
)

yoloV3model.fit(1)
