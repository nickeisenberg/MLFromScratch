import json
import os
from PIL import Image, ImageDraw

DATAPATH = "/home/nicholas/Datasets/flir/images_thermal_train"

with open(os.path.join(DATAPATH, 'coco.json'), 'r') as file:
    data = json.load(file)

print(data.keys())

data['annotations'][0]

img_id = data['images'][0]['id']
img_path = os.path.join(DATAPATH, data['images'][0]['file_name'])
img_annotes = [x for x in data['annotations'] if x['image_id'] == img_id]

img = Image.open(img_path)

if img.mode == 'L':
    rgb_img = Image.new("RGB", img.size)
    rgb_img.paste(img)
    img = rgb_img

draw = ImageDraw.Draw(img)

for img_a in img_annotes[0:1]:
    bbox = img_a['bbox']

    print(bbox)

    for i in [0]:
        bottom_right_x = bbox[0] + i + bbox[2]
        bottom_right_y = bbox[1] + i + bbox[3]
        draw.rectangle(
            (bbox[0] + i, bbox[1] + i, bottom_right_x, bottom_right_y), 
            outline ="red"
        )

img.show()
