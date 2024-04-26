import torch
import torch.nn as nn


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self):
        super().__init__()
        self.temp = 0.05
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


if __name__ == "__main__":
    item_emb = torch.ones([1, 10, 3])  # 1 x 10 x 3
    pooler = torch.ones([2, 1, 3])  # 2 x 3
    sim = Similarity()
    a = sim(pooler, item_emb)
    print(a.shape)
    print(a)
    labels = torch.Tensor()
    ce = nn.CrossEntropyLoss()
    l = ce(a, labels)
    print(l)
    print(l.shape)
