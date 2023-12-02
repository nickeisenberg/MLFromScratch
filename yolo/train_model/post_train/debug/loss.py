"""
I think obj.sum() == 0 may be causing problems. Going to investiate how frequent 
this occurs with my current batch size of 24.
"""

from train_model.settings import t_dataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

dataloader = DataLoader(t_dataset, 24, False)

bar = tqdm(dataloader, leave=True)

count = 0
for i, batch in enumerate(bar):
    imgs, targets = batch
    for t_s in targets:
        dims = torch.where(t_s[..., 4: 5] == 1)
        if dims[0].size()[0] > 0:
            continue
        else:
            count += 1
        bar.set_postfix(count=count)

"""
Here are the results. There are 38 times where obj.sum() = 0.

100%|█████████████████████████████████████████| 428/428 [05:16<00:00,  1.35it/s, count=38]
"""
