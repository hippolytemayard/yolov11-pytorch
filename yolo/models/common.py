"""
Common building blocks for YOLO v11.

This module contains reusable neural network components:
- Conv: Standard convolution + BatchNorm + activation
- Residual: Residual connection block
- CSP: Cross Stage Partial block
- SPP: Spatial Pyramid Pooling
- Attention: Self-attention mechanism
- PSA: Positional Self-Attention block
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """
    Fuse convolution and batch normalization layers for inference optimization.
    
    This technique merges Conv2d and BatchNorm2d into a single Conv2d layer,
    reducing memory access and computation during inference.
    
    Args:
        conv: Convolution layer to fuse
        bn: Batch normalization layer to fuse
        
    Returns:
        Fused convolution layer with bias
    """
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        groups=conv.groups,
        bias=True
    ).requires_grad_(False).to(conv.weight.device)

    # Fuse weights
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fused_conv.weight.copy_(torch.mm(w_bn, w_conv).view(fused_conv.weight.size()))

    # Fuse biases
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused_conv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fused_conv


class Conv(nn.Module):
    """
    Standard Convolution block: Conv2d + BatchNorm2d + Activation.
    
    This is the fundamental building block used throughout YOLO architecture.
    
    Args:
        in_ch: Number of input channels
        out_ch: Number of output channels
        activation: Activation function (e.g., nn.SiLU())
        k: Kernel size (default: 1)
        s: Stride (default: 1)
        p: Padding (default: 0)
        g: Groups for grouped convolution (default: 1)
    """
    
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        activation: nn.Module,
        k: int = 1,
        s: int = 1,
        p: int = 0,
        g: int = 1
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.norm = nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.03)
        self.relu = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with conv -> norm -> activation."""
        return self.relu(self.norm(self.conv(x)))

    def forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for fused conv (no separate norm)."""
        return self.relu(self.conv(x))


class Residual(nn.Module):
    """
    Residual block with bottleneck structure.
    
    Uses two 3x3 convolutions with a skip connection.
    The expansion ratio controls the bottleneck width.
    
    Args:
        ch: Number of input/output channels
        e: Expansion ratio for bottleneck (default: 0.5)
    """
    
    def __init__(self, ch: int, e: float = 0.5):
        super().__init__()
        hidden_ch = int(ch * e)
        self.conv1 = Conv(ch, hidden_ch, nn.SiLU(), k=3, p=1)
        self.conv2 = Conv(hidden_ch, ch, nn.SiLU(), k=3, p=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        return x + self.conv2(self.conv1(x))


class CSPModule(nn.Module):
    """
    Cross Stage Partial module with internal residual connections.
    
    This module splits features, processes one part through residuals,
    and concatenates with the other part for better gradient flow.
    
    Args:
        in_ch: Number of input channels
        out_ch: Number of output channels
    """
    
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        hidden_ch = out_ch // 2
        self.conv1 = Conv(in_ch, hidden_ch, nn.SiLU())
        self.conv2 = Conv(in_ch, hidden_ch, nn.SiLU())
        self.conv3 = Conv(2 * hidden_ch, out_ch, nn.SiLU())
        self.res_m = nn.Sequential(
            Residual(hidden_ch, e=1.0),
            Residual(hidden_ch, e=1.0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with CSP structure."""
        y = self.res_m(self.conv1(x))
        return self.conv3(torch.cat((y, self.conv2(x)), dim=1))


class CSP(nn.Module):
    """
    Cross Stage Partial block with configurable depth.
    
    This is the main CSP block used in the backbone and neck.
    It supports both simple residual and CSPModule variants.
    
    Args:
        in_ch: Number of input channels
        out_ch: Number of output channels
        n: Number of residual/CSP modules
        csp: If True, use CSPModule; else use simple Residual
        r: Channel reduction ratio
    """
    
    def __init__(self, in_ch: int, out_ch: int, n: int, csp: bool, r: int):
        super().__init__()
        hidden_ch = out_ch // r
        self.conv1 = Conv(in_ch, 2 * hidden_ch, nn.SiLU())
        self.conv2 = Conv((2 + n) * hidden_ch, out_ch, nn.SiLU())

        if not csp:
            self.res_m = nn.ModuleList(Residual(hidden_ch) for _ in range(n))
        else:
            self.res_m = nn.ModuleList(CSPModule(hidden_ch, hidden_ch) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with progressive feature concatenation."""
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv2(torch.cat(y, dim=1))


class SPP(nn.Module):
    """
    Spatial Pyramid Pooling block.
    
    Uses multiple max pooling operations at different scales to capture
    multi-scale context information.
    
    Args:
        in_ch: Number of input channels
        out_ch: Number of output channels
        k: Kernel size for max pooling (default: 5)
    """
    
    def __init__(self, in_ch: int, out_ch: int, k: int = 5):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 2, nn.SiLU())
        self.conv2 = Conv(in_ch * 2, out_ch, nn.SiLU())
        self.pool = nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with cascaded pooling."""
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))


class Attention(nn.Module):
    """
    Multi-head self-attention mechanism.
    
    Implements scaled dot-product attention with learnable positional
    encoding through depthwise convolution.
    
    Args:
        ch: Number of input channels
        num_head: Number of attention heads
    """
    
    def __init__(self, ch: int, num_head: int):
        super().__init__()
        self.num_head = num_head
        self.dim_head = ch // num_head
        self.dim_key = self.dim_head // 2
        self.scale = self.dim_key ** -0.5

        # QKV projection
        self.qkv = Conv(ch, ch + self.dim_key * num_head * 2, nn.Identity())
        
        # Positional encoding (depthwise conv)
        self.conv1 = Conv(ch, ch, nn.Identity(), k=3, p=1, g=ch)
        # Output projection
        self.conv2 = Conv(ch, ch, nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with multi-head attention."""
        b, c, h, w = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.view(b, self.num_head, self.dim_key * 2 + self.dim_head, h * w)
        q, k, v = qkv.split([self.dim_key, self.dim_key, self.dim_head], dim=2)

        # Scaled dot-product attention
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)

        # Apply attention and add positional encoding
        x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.conv1(v.reshape(b, c, h, w))
        return self.conv2(x)


class PSABlock(nn.Module):
    """
    Positional Self-Attention block.
    
    Combines self-attention with a feed-forward network (FFN)
    using residual connections.
    
    Args:
        ch: Number of channels
        num_head: Number of attention heads
    """
    
    def __init__(self, ch: int, num_head: int):
        super().__init__()
        self.conv1 = Attention(ch, num_head)
        self.conv2 = nn.Sequential(
            Conv(ch, ch * 2, nn.SiLU()),
            Conv(ch * 2, ch, nn.Identity())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention and FFN."""
        x = x + self.conv1(x)
        return x + self.conv2(x)


class PSA(nn.Module):
    """
    Positional Self-Attention module.
    
    Applies multiple PSA blocks to one branch of a split feature map,
    then concatenates with the other branch.
    
    Args:
        ch: Number of channels
        n: Number of PSA blocks
    """
    
    def __init__(self, ch: int, n: int):
        super().__init__()
        hidden_ch = ch // 2
        self.conv1 = Conv(ch, 2 * hidden_ch, nn.SiLU())
        self.conv2 = Conv(2 * hidden_ch, ch, nn.SiLU())
        self.res_m = nn.Sequential(*(PSABlock(hidden_ch, hidden_ch // 64) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with PSA."""
        x, y = self.conv1(x).chunk(2, 1)
        return self.conv2(torch.cat((x, self.res_m(y)), dim=1))


class Proto(nn.Module):
    """
    Prototype generation module for instance segmentation.
    
    Generates mask prototypes that are combined with mask coefficients
    from the detection head to produce instance masks.
    
    Args:
        in_ch: Number of input channels
        proto_ch: Number of prototype channels (internal)
        out_ch: Number of output mask prototypes
    """
    
    def __init__(self, in_ch: int, proto_ch: int = 256, out_ch: int = 32):
        super().__init__()
        self.conv1 = Conv(in_ch, proto_ch, nn.SiLU(), k=3, p=1)
        self.upsample = nn.ConvTranspose2d(proto_ch, proto_ch, 2, stride=2, bias=True)
        self.conv2 = Conv(proto_ch, proto_ch, nn.SiLU(), k=3, p=1)
        self.conv3 = Conv(proto_ch, out_ch, nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate mask prototypes."""
        x = self.conv1(x)
        x = self.upsample(x)
        x = self.conv2(x)
        return self.conv3(x)

