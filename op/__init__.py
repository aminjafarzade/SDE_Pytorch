# Lightweight, pure-PyTorch versions of the StyleGAN2 ops used in this project.
# We avoid compiling any custom C++/CUDA extensions so the code runs cleanly on
# clusters where JIT compilation is problematic.

import torch
import torch.nn.functional as F

from .upfirdn2d import upfirdn2d  # our pure PyTorch implementation


def fused_leaky_relu(x, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    """Simple PyTorch version of fused_leaky_relu.

    This is not as optimized as the custom CUDA op, but it has the same API and
    is perfectly fine for our purposes.
    """
    if bias is not None:
        # Assume NCHW and bias of shape [C]
        x = x + bias.view(1, -1, 1, 1)
    return F.leaky_relu(x, negative_slope) * scale


class FusedLeakyReLU(torch.nn.Module):
    """Module wrapper for fused_leaky_relu with a compatible interface."""

    def __init__(self, channels=None, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.negative_slope = negative_slope
        self.scale = scale
        # In the original implementation, bias is often handled outside.
        # If needed, it can be added as a Parameter here.

    def forward(self, x):
        return fused_leaky_relu(
            x,
            bias=None,
            negative_slope=self.negative_slope,
            scale=self.scale,
        )


__all__ = ["upfirdn2d", "fused_leaky_relu", "FusedLeakyReLU"]
