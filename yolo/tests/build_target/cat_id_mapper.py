from utils import Dataset, BuildTarget
import os
import json
from torchvision.transforms import v2
import numpy as np
from torch.utils.data import DataLoader
import torch

TRAINROOT = os.path.join(
    os.environ['HOME'], 'Datasets', 'flir', 'images_thermal_train'
)
ANNOT_FILE_PATH = os.path.join(
    TRAINROOT , 'coco.json'
)
with open(ANNOT_FILE_PATH, 'r') as oj:
    annotations = json.load(oj)

annotations['images'][0]
annotations['annotations'][0]

annotations.keys()

class CategoryMapper:
    def __init__(self, annotations, instructions={}):
        self.annotations = annotations 
        self.instructions = instructions
        self._annotations()
        self._categories()
        if len(instructions) > 0:
            self._transform_annotations()
        self._category_mapper()

    def _annotations(self):
        if isinstance(self.annotations, str):
            with open(self.annotations, 'r') as oj:
                self.annotations = json.load(oj)
        elif isinstance(self.annotations, dict):
            self.annotations = self.annotations
        else:
            print("annotations wernt loaded")
            return None

    def _categories(self):
        self.category_id = {}
        self.id_category = {}
        for x in self.annotations['annotations']:
            cat_id  = x['category_id']

            for catinfo in self.annotations['categories']:
                if catinfo['id'] == cat_id:
                    if cat_id not in self.id_category.keys():
                        self.category_id[catinfo['name']] = cat_id
                        self.id_category[cat_id] = catinfo['name']

    def _transform_annotations(self):
        new_annotations = {'images': [], 'annotations': [], 'categories': []}
        image_ids = []
        for annote in self.annotations['annotations']:
            annote_cat_id = annote['category_id']
            annote_cat = self.id_category[annote_cat_id]
            if annote_cat in self.instructions.keys():
                if self.instructions[annote_cat] == 'ignore':
                    continue
                else:
                    new_annotations['annotations'].append(annote)
                    image_ids.append(annote['image_id'])
            else:
                new_annotations['annotations'].append(annote)
                image_ids.append(annote['image_id'])

        for image in self.annotations['images']:
            if image['id'] not in image_ids:
                continue
            else:
                new_annotations['images'].append(image)

        for category in self.annotations['categories']:
            if category['name'] in self.instructions.keys():
                if self.instructions[category['name']] == 'ignore':
                    continue
                else:
                    new_annotations['categories'].append(
                        {
                            'id': category['id'], 
                            'name': self.instructions[category['name']], 
                        }
                    )
            else:
                new_annotations['categories'].append(category)

        self.annotations = new_annotations

    def _category_mapper(self):
        self.cat_mapper = {}
        for category in self.annotations['categories']:
            self.cat_mapper['id'] = len(self.cat_mapper)

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

cat_mapper = CategoryMapper(annotations, instructions)

cat_mapper = CategoryMapper(annotations)

cat_mapper.category_id

cat_mapper.id_category

len(cat_mapper.annotations['categories'])

cat_mapper.cat_mapper


