from trfc.dataset.objdet.utils.yolo import ConstructAnchors, YOLOTarget
from trfc.dataset.objdet import COCODataset
from trfc.dataset.objdet.utils import COCOjsonTransformer

path_to_coco = "/home/nicholas/Datasets/flir/images_thermal_train/coco.json"

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

coco = COCOjsonTransformer(path_to_coco, instructions, 40, 40)

class_names = coco.class_names
class_id_mapper = coco.class_id_mapper

anchors = ConstructAnchors(coco.coco, 640, 512).anchors[:, 1:]

scales = [32, 16, 8]

yoloTarget = YOLOTarget(class_id_mapper, anchors, scales, 640, 512)

coco_dataset = COCODataset(
    coco.coco, 
    "coco_dataset",
    class_names=class_names,
    class_id_mapper=class_id_mapper,
    fix_file_path="/home/nicholas/Datasets/flir/images_thermal_train",
    target_transform=yoloTarget.build
)
