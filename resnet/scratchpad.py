import torch
import torch.nn as nn
import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Loading a image to manipulate
celebaimgsdir = "/home/nicholas/Datasets/celebA/imgs"
celebaimgs = os.listdir(celebaimgsdir)
img_pil = Image.open(os.path.join(celebaimgsdir, celebaimgs[0]))
transform = transforms.Compose(
    [
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
    ]
)
img = transform(img_pil)
imgs = torch.unsqueeze(img, 0)


# kernal 1 and stride 1 conv2d layer.
conv = nn.Conv2d(3, 3, 1, 1)

img_t_np = conv(img).detach().numpy()
img_np = img.detach().numpy()

fig, ax = plt.subplots(1, 2)
ax[0].imshow(img_np[0])
ax[1].imshow(img_t_np[0])
plt.show()


# initial layer of resnet
initial = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

nn.AdaptiveMaxPool2d((1, 1))(torch.randn((32, 128, 7, 2))).shape

nn.Flatten()(torch.randn((32, 128, 7, 2))).shape
