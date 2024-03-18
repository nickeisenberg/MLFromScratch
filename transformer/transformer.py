import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, k, heads=4, mask=False) -> None:
        super().__init__()
        assert k % heads == 0

        self.k, self.heads = k, heads

        self.tokeys = nn.Linear(k, k, bias=False)
        self.toqueries = nn.Linear(k, k, bias=False)
        self.tovalues = nn.Linear(k, k, bias=False)
        self.unifyheads = nn.Linear(k, k)

    def forward(self, x: torch.Tensor):
        b, t, k = x.size()
        s = k // self.heads

        queries = self.toqueries(x)
        keys = self.tokeys(x)
        values = self.tovalues(x)

        keys = keys.transpose(1, 2).contiguous().view(b * self.heads, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * self.heads, t, s)
        values = values.transpose(1, 2).contiguous().view(b * self.heads, t, s)

        dot = torch.bmm(queries, keys.transpose(1, 2))
        dot = dot / (k ** (1/2))
        dot = F.softmax(dot, dim=2)
        out = torch.bmm(dot, values).view(b, self.heads, t, s)
        out = out.transpose(1, 2).contiguous().view(b, t, s * self.heads)

        return self.unifyheads(out)


class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()

        self.attention = SelfAttention(k, heads=heads)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k)
        )

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)
        fedforward = self.ff(x)
        return self.norm2(fedforward + x)


class Transformer(nn.Module):
    def __init__(self, k, heads, depth, seq_length, num_tokens, num_classes):
        super().__init__()

        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, k)
        self.pos_emb = nn.Embedding(seq_length, k)

        tblocks = []
        for _ in range(depth):
            tblocks.append(TransformerBlock(k=k, heads=heads))
        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(k, num_classes)

    def forward(self, x):
        """
        :param x: A (b, t) tensor of integer values representing
                  words (in some predetermined vocabulary).
        :return: A (b, c) tensor of log-probabilities over the
                 classes (where c is the nr. of classes).
        """
        tokens = self.token_emb(x)
        b, t, k = tokens.size()
        
        positions = torch.arange(t)
        positions = self.pos_emb(positions).unsqueeze(0).expand(b, t, k)
        
        x = tokens + positions
        x = self.tblocks(x)
        
        x = self.toprobs(x.mean(dim=1))
        return F.log_softmax(x, dim=1)


transformer = Transformer(12, 4, 2, 25, 400, 5)
transformer(torch.randint(0, 400, (33, 25))).shape
