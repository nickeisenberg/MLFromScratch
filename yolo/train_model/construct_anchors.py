import os
import json
from utils import ConstructAnchors
from train_model.settings import t_at

annotations = t_at.annotations

construct_anchors = ConstructAnchors(annotations['annotations'], 640, 512)

print(construct_anchors.cluster_centers[:, 1:])
