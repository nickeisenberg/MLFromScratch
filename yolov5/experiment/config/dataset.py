from trfc.dataset.objdet.utils.yolo import ConstructAnchors, YOLOTarget
from trfc.dataset.objdet import COCODataset
from trfc.dataset.objdet.utils import COCOjsonTransformer
from trfc.dataset.base import DataLoader
from torchvision.transforms import transforms
from PIL import Image
import torch

train = "/home/nicholas/Datasets/flir/images_thermal_train/coco.json"
val = "/home/nicholas/Datasets/flir/images_thermal_val/coco.json"

instructions = {
    "light": "ignore",
    "sign": "ignore",
    "person": "ignore",
    "bike": "ignore",
    "hydrant": "ignore",
    "deer": "ignore",
    "skateboard": "ignore",
    "train": "ignore",
    "dog": "ignore",
    "stroller": "ignore",
    "scooter": "ignore",
}

coco_train = COCOjsonTransformer(train, instructions, 40, 40)
coco_val = COCOjsonTransformer(val, instructions, 40, 40)

class_names = coco_train.class_names
class_id_mapper = coco_train.class_id_mapper

anchors = ConstructAnchors(coco_train.coco, 640, 512).anchors[:, 1:]
scales = torch.tensor([32, 16, 8])

yoloTarget = YOLOTarget(class_id_mapper, anchors, scales, 640, 512)

target_transform = yoloTarget.build

import torch

_img_transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def img_transform(path, tform=_img_transform):
    return tform(Image.open(path))

train_dataset = COCODataset(
    coco_train.coco, 
    "train_dataset",
    class_names=class_names,
    class_id_mapper=class_id_mapper,
    fix_file_path="/home/nicholas/Datasets/flir/images_thermal_train",
    target_transform=target_transform,
    img_transform=img_transform
)

val_dataset = COCODataset(
    coco_val.coco, 
    "val_dataset",
    class_names=class_names,
    class_id_mapper=class_id_mapper,
    fix_file_path="/home/nicholas/Datasets/flir/images_thermal_train",
    target_transform=target_transform,
    img_transform=img_transform
)

train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=False)
