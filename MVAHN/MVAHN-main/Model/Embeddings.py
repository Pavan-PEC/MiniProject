import einops
import torch
import math
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from einops.layers.torch import Rearrange

class PatchEmbeddings(nn.Module):
"""
 For converting the input image into 1-D array
"""
    def __init__(self, patch_size: int, patch_dim: int, emb_dim: int):
        # patch_size: dimensions of patch, patch_dim: No. of values present in a patch, emb_dim: No. of values after applying linear transformation on patches
        super().__init__()
        self.patchify = Rearrange(
            "b c (h p1) (w p2) -> b (h w) c p1 p2",
            p1=patch_size, p2=patch_size)
        # input image dimensions to patch dimensions
        # batch_size, channels(colors), height*patch_size, width*patch_suze
        # batch_size, height*width, c, patch_size, patch_size

        self.flatten = nn.Flatten(start_dim=2) # multi-dim array to 1-D array
        self.proj = nn.Linear(in_features=patch_dim, out_features=emb_dim) # returns embedded values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patchify(x)
        x = self.flatten(x)
        x = self.proj(x)

        return x


class PositionalEmbeddings(nn.Module):
    """
    For getting Embeddings (1-D array+patch position)
    """

    def __init__(self, num_pos: int, dim: int):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(num_pos, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos