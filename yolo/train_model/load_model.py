import torch
from model import YoloV3
from utils import Dataset
import os
import json
from torchvision.transforms import v2
import matplotlib.pyplot as plt

scales = torch.tensor([32, 16, 8])

yolo = YoloV3(1, scales, 7)

sd = "/home/nicholas/GitRepos/ml_arcs/yolo/train_model/state_dicts/yolo_1.pth"

yolo.load_state_dict(torch.load(sd))


TRAINROOT = os.path.join(
    os.environ['HOME'], 'Datasets', 'flir', 'images_thermal_train'
)
ANNOT_FILE_PATH = os.path.join(
    TRAINROOT , 'coco.json'
)
with open(ANNOT_FILE_PATH, 'r') as oj:
    annotations = json.load(oj)

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

# Input transform
img_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

dataset = Dataset(
    annot_json=annotations,
    annot_image_key='images', 
    annot_bbox_key='annotations', 
    image_file_name='file_name', 
    image_image_id='id', 
    bbox_bbox='bbox', 
    bbox_image_id='image_id', 
    bbox_category_id='category_id', 
    img_transform=img_transform,
    fix_file_path=TRAINROOT
)

im, tar = dataset.__getitem__(0)


x = yolo(im.unsqueeze(0))[2][..., 4: 5].reshape(-1).detach().numpy()
plt.plot(x)
plt.show()
