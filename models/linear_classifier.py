import torch
from torch import nn


class LinearClassifier(nn.Module):
    def __init__(self, emb_size: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(emb_size, num_classes)

    def forward(self, embeddings: torch.Tensor):
        return self.linear(embeddings)
