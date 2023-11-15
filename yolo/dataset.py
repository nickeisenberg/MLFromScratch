from torch.utils.data import Dataset as _Dataset
import os
import json
from PIL import Image


class Dataset(_Dataset):
    """
    A general YOLO Dataset.
    The annotation json file are required to have the following keys:
        1. "<image>"
        2. "<bbox>"
    These two keys should have a list of dictionaries as their values. Each
    dictionary will contatin metadata relating to the image and bbxom
    respectively. Requried in these dictionaries is the following:
        1. The "<image>" key values must contain
            1.1) "<file_name>": path to the image.
            1.2) "<image_id>": unique image id.
        2. The "<bbox>" key values must contain
            2.1) "<bbox>": [bx, by, bw, bh] where (bx, by) are the coords of
                 the top left of the image and bw amd bh are the width and
                 height.
            2.2) "<image_id>":  The unique image id number.
            2.3) "<category_id>": The unique category id of the image in the 
                 bbox.
    
    """

    def __init__(self, annot_file_path, annot_image_key, annot_bbox_key,
                 image_file_name, image_image_id, bbox_bbox, bbox_image_id,
                 bbox_category_id, transforms=None, fix_file_path=None):

        with open(annot_file_path, 'r') as oaf:
            self.annot = json.load(oaf)
        
        # The annot file keys
        self.annot_image_key = annot_image_key
        self.annort_bbox_key = annot_bbox_key
        self.image_file_name = image_file_name
        self.image_image_id = image_image_id
        self.bbox_bbox = bbox_bbox
        self.bbox_image_id = bbox_image_id
        self.bbox_category_id = bbox_category_id
        #--------------------------------------------------
        
        # optionals
        self.transforms = transforms
        self.fix_file_path = fix_file_path
        #--------------------------------------------------

    def __len__(self):
        return len(self.annot[self.annot_image_key])

    def __getitem__(self, idx):
        img_path = self.annot[self.annot_image_key][idx][self.image_file_name]
        if self.fix_file_path:
            img_path = os.path.join(self.fix_file_path, img_path)

        img = Image.open(img_path)
        if self.transforms:
            img = self.transforms(img)

        img_id = self.annot[self.annot_image_key][idx][self.image_image_id]
        img_annots = [
            x for x in self.annot[self.annort_bbox_key] 
            if x[self.bbox_image_id] == img_id
        ]

        return img, img_annots

    def _make_targets(self, img_annots):
        return None


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

dataset = Dataset(
    ANNOT_FILE_PATH, 'images', 'annotations', 'file_name', 'id', 'bbox', 
    'image_id', 'category_id', fix_file_path=TRAINROOT
)

for im, a in dataset:
    im.show()
    a[0]
    break
