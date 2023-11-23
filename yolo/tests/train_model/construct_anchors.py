import os
import json
from utils import ConstructAnchors

trainroot = os.path.join(
    os.environ['HOME'], 'Datasets', 'flir', 'images_thermal_train'
)
annote_file_path = os.path.join(
    trainroot , 'coco.json'
)

with open(annote_file_path, 'r') as oj:
    annotations = json.load(oj)

construct_anchors = ConstructAnchors(annotations['annotations'], 640, 512)
print(construct_anchors.cluster_centers[:, 1:])
