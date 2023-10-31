import torch
from torch.utils.data import random_split
from model import ResNet 
import numpy as np
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


def train_one_epoch(epoch):
   
    epoch_loss = 0
    running_loss = 0
    batch_num = 1

    for batch_num, (inputs, labels) in enumerate(train_dataloader):

        if (batch_num + 1) % 20 == 0:
            print(f"EPOCH {epoch} LOSS {np.round(running_loss / 20, 3)}")
            running_loss = 0

        inputs = inputs.to(device)
        labels = labels.to(device)
    
        optimizer.zero_grad()
        outputs = torch.max(resnet(inputs), dim=-1 keepdim=False).values
        loss = loss_fn(outputs, labels)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        running_loss += loss.item()

    avg_epoch_loss = np.round(epoch_loss / batch_num, 3)

    with torch.no_grad():
        val_loss = 0
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = torch.max(resnet(inputs), dim=-1, keepdim=False).values
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()

    avg_val_loss = np.round(val_loss / len(val_dataloader), 3)

    print(f"EPOCH LOSS {avg_epoch_loss} VALIDATION LOSS {avg_val_loss}")

    return avg_epoch_loss, avg_val_loss


imgs = torch.randn((20000, 3, 128, 128))
labels = torch.randint(0, 10, (20000,))

dataset = CustomImageDataset(image_data=imgs, labels=labels)

train_size = int(.8 * len(dataset))
val_size = len(dataset) - int(.8 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 128
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

resnet = ResNet(10)
resnet.to(device)
optimizer = torch.optim.RMSprop(resnet.parameters())
loss_fn = torch.nn.MSELoss()

for i in range(3):
    epoch_loss, val_loss = train_one_epoch(i + 1)
