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

    Notes
    -----

    #-------------------- 
    # example file structure
    #-------------------- 
    /home/user
        - Datasets
            - coco_dir
                - coco_imgs
                    -img1.png
                    -img2.png
                    ...
                - coco.json

    #-------------------- 
    # coco.json example
    #-------------------- 
    coco_json = {
        "image": [
            {
                "file_name": "coco_imgs/img1.png",
                "id": 0,
            },
            ...
        ],
        "bbox": [
            {
                "bbox": [x, y, w, h],
                "image_id": 0,
                "category_id": 0
            },
            ...
        ]
    }

    #-------------------- 
    # dataset inputs
    #-------------------- 
    dataset_kwargs = {
        annot_json: coco_json, 
        annot_image_key: "image", 
        annot_bbox_key: "bbox",
        image_file_name: "file_name", 
        image_image_id: "id", 
        bbox_bbox: "bbox", 
        bbox_image_id: "image_id",
        bbox_category_id: "category_id", 
        img_transform: None, 
        target_transform: None, 
        fix_file_path: "/home/user/Datasets/coco_dir"
    }

    #-------------------- 
    # output of the dataset
    #-------------------- 
    dataset = Dataset(**dataset_kwargs)

    dataset[0] = (img_path, annote_list)

    img_path = "/home/user/Datasets/coco_dir/coco_imgs/img1.png"

    annote_list = [
        {"bbox": [x, y, w, h], "image_id": 0, "category_id": 0},
        {"bbox": [x, y, w ,h], "image_id": 0, "category_id": 1},
        ...
    ]
    
    """

    def __init__(self, annot_json, annot_image_key, annot_bbox_key,
                 image_file_name, image_image_id, bbox_bbox, bbox_image_id,
                 bbox_category_id, img_transform=None, target_transform=None, 
                 fix_file_path=None):

        if isinstance(annot_json, str): 
            with open(annot_json, 'r') as oaf:
                self.annot = json.load(oaf)
        elif isinstance(annot_json, dict):
            self.annot = annot_json
        
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
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.fix_file_path = fix_file_path
        #--------------------------------------------------

    def __len__(self):
        return len(self.annot[self.annot_image_key])

    def __getitem__(self, idx):
        img_path = self.annot[self.annot_image_key][idx][self.image_file_name]
        if self.fix_file_path:
            img_path = os.path.join(self.fix_file_path, img_path)

        img = Image.open(img_path)
        if self.img_transform:
            img = self.img_transform(img)

        img_id = self.annot[self.annot_image_key][idx][self.image_image_id]

        img_annots = [
            x for x in self.annot[self.annort_bbox_key] 
            if x[self.bbox_image_id] == img_id
        ]
        if self.target_transform:
            img_annots = self.target_transform(img_annots)

        return img, img_annots
