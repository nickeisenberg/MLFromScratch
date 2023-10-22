import torch
import torch.nn as nn

input = torch.randn(size=(16, 3, 64, 64))

#--------------------------------------------------
# block in encode
#--------------------------------------------------
conv1 = nn.Conv2d(
    3,
    16,
    stride=1,
    padding=1,
    kernel_size=3
)
pooling = nn.MaxPool2d((2, 2))
output = pooling(conv1(input))
output.shape

#--------------------------------------------------
# block in decode
#--------------------------------------------------
convt1 = nn.ConvTranspose2d(
    16,
    3,
    stride=2,
    padding=0,
    kernel_size=2
)
convt1(output).shape

conv = nn.Conv2d(
    3 + 3,
    3,
    stride=1,
    padding=1,
    kernel_size=3
)

conv(
    torch.cat(
        (input, convt1(output)),
        axis=1
    )
).shape
