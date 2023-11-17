import json
import os
from utils import ConstructAnchors


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


construct_anchors = ConstructAnchors(annotations['annotations'], 640, 512)

construct_anchors.view_clusters()

construct_anchors.cluster_centers
