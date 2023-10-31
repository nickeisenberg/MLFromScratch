import torch
from model import ResNet 
from torch.utils.data import Dataset, DataLoader


device = "cuda" if torch.cuda.is_available() else "cpu"

class CustomImageDataset(Dataset):
    def __init__(self, image_data, labels, transform=None):
        self.image_data = image_data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        img = torch.tensor(self.image_data[idx], dtype=torch.float32)
        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return img, label


imgs = torch.randn((20000, 3, 128, 128))
labels = torch.randint(0, 10, (20000,))

dataset = CustomImageDataset(image_data=imgs, labels=labels)
batch_size = 128
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

resnet = ResNet(10)
resnet.to(device)

for i, (input, label) in enumerate(dataloader):
    if i % 20 == 0:
        print(f"{i} / {len(dataloader)}")
    input = input.to(device)
    label = label.to(device)
    output = resnet(input)
