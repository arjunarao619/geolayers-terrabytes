from typing import Any, Callable, Dict

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


class MinMaxNormalize(torch.nn.Module):
    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        rgb = x["image"][:3, :, :]  # First 3 bands are RGB
        min_val = torch.min(rgb)
        max_val = torch.max(rgb)
        # Avoid division by zero
        rgb = (rgb - min_val) / (max_val - min_val + 1e-5)
        x["image"][:3, :, :] = rgb
        return x
