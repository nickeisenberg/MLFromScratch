import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, k, heads=4, mask=False) -> None:
        super().__init__()
        assert k % heads == 0

        self.k, self.heads = k, heads

        self.tokeys = nn.Linear(k, k, bias=False)
        self.toqueries = nn.Linear(k, k, bias=False)
        self.tovalues = nn.Linear(k, k, bias=False)
        self.unifyheads = nn.Linear(k, k, bias=False)



