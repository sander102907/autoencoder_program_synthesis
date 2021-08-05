import torch.nn.functional as F
import torch.nn as nn
import torch


class CosineDistance(nn.Module):
    """
        Returns cosine distance between x1 and x2 computed along dim. 
        Uses torch.nn.functional.cosine_similarity to calculate cosine similarity.
        The cosine distance is then defined as 1 - cosine similarity.
    """

    def __init__(self, dim: int =1, eps: float = 1e-8) -> None:
        super().__init__()

        self.dim = dim
        self.eps = eps

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return 1.0 - F.cosine_similarity(x1, x2, dim=self.dim, eps=self.eps)
