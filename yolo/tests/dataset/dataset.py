from utils import Dataset
import os
import json
from torchvision.transforms import v2
import torch

HOME = os.environ['HOME']
FLIRROOT = os.path.join(
    HOME, 'Datasets', 'flir'
)
TRAINROOT = os.path.join(
    FLIRROOT, 'images_thermal_train'
)
ANNOT_FILE_PATH = os.path.join(
    TRAINROOT , 'coco.json'
)
with open(ANNOT_FILE_PATH, 'r') as oj:
    annotations = json.load(oj)

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

dataset = Dataset(
    ANNOT_FILE_PATH, 'images', 'annotations', 'file_name', 'id', 'bbox', 
    'image_id', 'category_id', transforms=transform, fix_file_path=TRAINROOT
)

for im, a in dataset:
    im.shape
    a[0]
    break
