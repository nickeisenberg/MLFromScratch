"""
I will work on this later.
"""

import json
import os
from PIL import Image, ImageDraw

DATAPATH = "/home/nicholas/Datasets/flir/images_thermal_train"

with open(os.path.join(DATAPATH, 'coco.json'), 'r') as file:
    data = json.load(file)

#--------------------------------------------------
# Select an image to view the bounding boxes
#--------------------------------------------------
img_id = data['images'][90]['id']
#--------------------------------------------------

#--------------------------------------------------
# Get the image path and subset the annotations only to this image
#--------------------------------------------------
img_path = os.path.join(DATAPATH, data['images'][img_id]['file_name'])
img_annotes = [x for x in data['annotations'] if x['image_id'] == img_id]
#--------------------------------------------------

#--------------------------------------------------
# Open the image with PIL and add the bounding boxes and show the image
#--------------------------------------------------
img = Image.open(img_path)

if img.mode == 'L':
    rgb_img = Image.new("RGB", img.size)
    rgb_img.paste(img)
    img = rgb_img

draw = ImageDraw.Draw(img)

for img_a in img_annotes:
    bbox = img_a['bbox']
    for i in [0]:
        bottom_right_x = bbox[0] + i + bbox[2]
        bottom_right_y = bbox[1] + i + bbox[3]
        draw.rectangle(
            (bbox[0] + i, bbox[1] + i, bottom_right_x, bottom_right_y), 
            outline ="red"
        )

img.show()

def
#--------------------------------------------------
